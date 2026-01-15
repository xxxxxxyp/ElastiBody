import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os
from .sensitivity import SensitivityBuilder
from .regularization import Regularizer
from ..models.nhookean import NHookeanForwardSolver

class InverseSolver:
    def __init__(self, initializer, obs_step_idx=9):
        self.init = initializer
        self.cpp = initializer.cpp_backend
        self.step_idx = obs_step_idx
        self.model_name = initializer.model_name
        
        # 加载观测数据
        data_dir = f"data-sampling/{self.model_name}"
        pnt_path = os.path.join(data_dir, f"pnt{obs_step_idx}.txt")
        force_path = os.path.join(data_dir, f"force{obs_step_idx}.txt")
        
        if not os.path.exists(pnt_path):
            raise FileNotFoundError(f"Observation data {pnt_path} not found.")
            
        self.nodes_obs = np.loadtxt(pnt_path)
        self.u_meas_full = self.nodes_obs - self.init.nodes.flatten()
        self.f_ext = np.loadtxt(force_path)
        
        # 表面节点用于填补步骤的 Dirichlet BC
        self.surface_indices = self._find_boundary_nodes()
        
        # 工具模块
        self.sensitivity_builder = SensitivityBuilder(initializer)
        self.regularizer = Regularizer(initializer.cells)
        self.L_matrix = self.regularizer.build_laplacian()
        self.current_E = np.full(initializer.num_cells, initializer.E_base)

    def _find_boundary_nodes(self):
        nodes = self.init.nodes
        min_xyz = np.min(nodes, axis=0)
        max_xyz = np.max(nodes, axis=0)
        tol = 1e-5
        indices = []
        for i, p in enumerate(nodes):
            if (np.any(np.abs(p - min_xyz) < tol) or 
                np.any(np.abs(p - max_xyz) < tol)):
                indices.append(i)
        return np.array(indices)

    def solve_alternating(self, lambda_reg=1e-8, max_iter=10, ignore_nodes=None):
        """
        交替迭代求解 (带灵敏度加权/雅可比预条件化)
        :param lambda_reg: 正则化系数
        :param ignore_nodes: 需要从线性方程组中剔除的节点索引列表 (如底面固定点)
        """
        print(f"\n=== Starting Inverse Optimization (Reg={lambda_reg}) ===")
        
        # 1. 构建有效方程的掩码 (Mask) - 用于剔除支座反力节点
        valid_dof_mask = np.ones(3 * self.init.num_nodes, dtype=bool)
        if ignore_nodes is not None:
            print(f"[Inverse] Excluding {len(ignore_nodes)} fixed nodes from force balance equations.")
            for idx in ignore_nodes:
                valid_dof_mask[3*idx : 3*idx+3] = False
        
        # 转换 f_ext 为 masked
        f_ext_masked = self.f_ext[valid_dof_mask]

        for k in range(max_iter):
            print(f"\n--- Iteration {k} ---")
            
            # =========================================================
            # Step 1: 填补缺失数据 (Imputation)
            # =========================================================
            print("1. Predicting internal displacements...")
            self.init.E_field = self.current_E
            self.init.commit_to_cpp()
            
            # 实例化前向求解器进行填补
            fwd_solver = NHookeanForwardSolver(self.init)
            # 固定所有表面节点为观测值
            fwd_solver.set_dirichlet_bc(self.surface_indices)
            fwd_solver.u = self.u_meas_full.copy()
            
            # 求解内部平衡 (step_index=999 防止覆盖文件)
            # 注意: 这里 f_ext 对应内部节点的体利(通常为0)和表面节点的力(被BC覆盖)
            fwd_solver.solve_static_step(self.f_ext, step_index=999, tol=1e-6)
            u_full_k = fwd_solver.u
            
            # =========================================================
            # Step 2: 构建与加权灵敏度矩阵 (Assembly & Weighting)
            # =========================================================
            print("2. Building & Weighting Sensitivity Matrix...")
            self.cpp.set_current_displacement(u_full_k)
            S_full = self.sensitivity_builder.build_sensitivity_matrix()
            
            # 剔除无效行 (Reaction Forces)
            S_valid = S_full[valid_dof_mask, :]
            
            # --- [核心修改] 灵敏度加权 (Jacobian Preconditioning) ---
            # 目的: 放大深层单元的梯度，压制表层单元的梯度，消除"趋肤效应"
            
            # 1. 计算每列范数 (每个单元的灵敏度强度)
            # S_valid 是 CSC 格式，计算列范数
            sensitivity_magnitude = sp.linalg.norm(S_valid, axis=0)
            
            # 2. 计算权重 W
            # 公式: w_j = 1 / ( ||S_j||^alpha + epsilon )
            epsilon = 1e-8 * np.max(sensitivity_magnitude)
            alpha = 1.5  # 调节因子 (0.0=无加权, 1.0=完全归一化). 0.6~0.8 通常效果好
            
            raw_weights = 1.0 / (sensitivity_magnitude**alpha + epsilon)
            
            # 归一化权重，使其均值为 1.0，避免干扰正则化项的量级
            weights = raw_weights / np.mean(raw_weights)
            
            # 构建对角矩阵 W
            W_diag = sp.diags(weights)
            
            # 3. 变换矩阵 S_tilde = S * W
            # 这样 S_weighted 的每一列范数都比较接近
            S_weighted = S_valid @ W_diag
            
            # =========================================================
            # Step 3: 求解加权线性系统 (Weighted Inversion)
            # =========================================================
            print("3. Solving weighted linear system...")
            
            # 目标方程: (S_w^T S_w + reg * L_w^T L_w) * E_tilde = S_w^T * f
            # 其中 E = W * E_tilde
            
            # 正则化矩阵也需要变换: L_w = L * W
            # 物理含义: 对于灵敏度低的区域(W大)，正则化约束(平滑要求)也被放大了，这是合理的
            L_weighted = self.L_matrix @ W_diag
            
            # 构建最小二乘系统
            AtA = S_weighted.T @ S_weighted
            Reg = lambda_reg * (L_weighted.T @ L_weighted)
            A_sys = AtA + Reg
            b_sys = S_weighted.T @ f_ext_masked
            
            try:
                # 求解无量纲变量 E_tilde
                E_tilde = spla.spsolve(A_sys, b_sys)
            except Exception as e:
                print(f"[Error] Linear solve failed: {e}")
                break
            
            # =========================================================
            # Step 4: 还原与约束 (Recovery & Constraints)
            # =========================================================
            
            # 1. 还原物理量: E = W * E_tilde
            E_new = weights * E_tilde
            
            # 2. 符号自动修正 (Auto Sign Fix)
            if np.mean(E_new) < 0:
                print("  [Info] Flipping sign...")
                E_new = -E_new
                
            # 3. 物理下限约束
            E_new = np.maximum(E_new, 1000.0) 
            
            # 计算变化率
            diff = np.linalg.norm(E_new - self.current_E) / np.linalg.norm(self.current_E)
            print(f"  E change ratio: {diff:.4f}")
            print(f"  Current Mean E: {np.mean(E_new):.2f}")
            
            self.current_E = E_new
            
            if diff < 0.01:
                print("-> Converged.")
                break
                
        return self.current_E
    def export_result(self, filename="we_recon.txt"):
        path = os.path.join("data-sampling", self.model_name, filename)
        np.savetxt(path, self.current_E)
        print(f"[IO] Reconstructed E saved to {path}")