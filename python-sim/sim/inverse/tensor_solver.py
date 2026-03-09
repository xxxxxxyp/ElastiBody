import numpy as np
import scipy.sparse as sp
from .adjoint_solver import AdjointSolver

class TensorIsotropicSolver(AdjointSolver):
    """
    基于 K 张量基底分解的反演求解器 (Tensor-based Isotropic Solver)
    
    特点：
    1. 不再依赖 gen_unit_geometric_forces (E的直接导数)
    2. 而是利用 gen_lame_sensitivities (mu, lambda 的导数)
    3. 在 Python 端通过链式法则投影回 E
    
    优势：未来只需修改投影公式，即可无缝切换到各向异性反演。
    """
    
    def __init__(self, initializer, u_obs_full, f_ext, fixed_dofs):
        # 复用父类的初始化逻辑
        super().__init__(initializer, u_obs_full, f_ext, fixed_dofs)
        
        # 预计算稀疏矩阵的索引结构 (Row, Col)，避免每次迭代重复计算
        # 这一步逻辑与 SensitivityBuilder 一致
        self._prepare_sparse_indices()

    def _prepare_sparse_indices(self):
        """预计算构建稀疏矩阵所需的行、列索引"""
        num_cells = self.init.num_cells
        cells = self.init.cells
        
        # 1. 列索引 (Cell Indices): [0...0, 1...1, ...] 每个重复12次
        self.col_indices = np.repeat(np.arange(num_cells), 12)
        
        # 2. 行索引 (Global DOF Indices)
        # 展开 cells: (N, 4) -> (N*4*3)
        # 顺序: Cell0_Node0_x, Cell0_Node0_y...
        cells_flat = cells.flatten() # [c0n0, c0n1, c0n2, c0n3, c1n0...]
        
        dof_indices = np.empty((num_cells * 4, 3), dtype=int)
        dof_indices[:, 0] = cells_flat * 3
        dof_indices[:, 1] = cells_flat * 3 + 1
        dof_indices[:, 2] = cells_flat * 3 + 2
        
        self.row_indices = dof_indices.flatten()
        
        # 记录维度
        self.num_dofs = self.init.num_nodes * 3
        self.num_cells = num_cells

    def _build_sparse_from_dense(self, dense_12xn):
        """
        辅助函数：将 C++ 返回的 (12, N) 稠密矩阵转换为 (3N, N) 稀疏矩阵 S
        """
        # 注意：C++ Eigen 是列优先，但我们填充数据时需要对应 row_indices 的顺序
        # row_indices 的顺序是：Cell0 的 12 个 DOF，Cell1 的 12 个 DOF...
        # dense_12xn 的列 0 是 Cell0 的数据
        # 因此我们需要 dense_12xn.T (变成 N x 12) 然后 flatten
        
        data_values = dense_12xn.T.flatten()
        
        S = sp.csc_matrix(
            (data_values, (self.row_indices, self.col_indices)), 
            shape=(self.num_dofs, self.num_cells)
        )
        return S

    def _objective_and_gradient(self, s_vec):
        """
        重写核心梯度计算逻辑：张量基底 -> 投影 -> 标量梯度
        """
        # 1. 参数映射 (Scalar E -> Physical E)
        E_physical = s_vec * self.E_scaling
        
        # 注入 C++ (用于前向仿真)
        self.cpp.set_element_modulus(E_physical)
        
        # 2. 前向仿真 (求 u_sim)
        try:
            # 复用父类的前向求解
            # 注意：如果父类 _solve_forward 是私有的，这里直接调用即可
            u_sim = self._solve_forward(E_physical)
        except Exception:
            # 仿真发散保护
            return 1e9, np.zeros_like(s_vec)
        
        # 3. 计算残差与 Loss
        diff = u_sim - self.u_obs
        diff[self.fixed_dofs] = 0.0
        loss = 0.5 * np.sum(diff**2)
        
        # 4. 伴随求解 (求 lambda)
        lam = self._solve_adjoint(u_sim, diff)
        
        # =====================================================
        # [核心差异]：从张量基底出发计算梯度
        # =====================================================
        
        # A. 调用 C++ 新接口，获取张量基底灵敏度
        # S_mu_dense, S_la_dense 都是 (12, N) 的矩阵
        S_mu_dense, S_la_dense = self.cpp.gen_lame_sensitivities()
        
        # B. 转为稀疏矩阵 (3N, N)
        S_mu = self._build_sparse_from_dense(S_mu_dense)
        S_la = self._build_sparse_from_dense(S_la_dense)
        
        # C. 计算本构参数梯度 (Target: mu, lambda)
        # grad_mu = S_mu^T * lambda
        grad_mu = S_mu.T @ lam  # Shape: (N_cells, )
        grad_la = S_la.T @ lam  # Shape: (N_cells, )
        
        # D. 投影层 (Projection Layer): (mu, lambda) -> E
        # 各向同性假设下的链式法则系数 (Jacobian)
        nu = 0.49 # 假设泊松比固定
        
        # E = mu * (3*la + 2*mu) / (la + mu) ... 公式太复杂
        # 不如反过来：mu = E / 2(1+v), la = E*v / (1+v)(1-2v)
        # dJ/dE = (dJ/dmu * dmu/dE) + (dJ/dla * dla/dE)
        
        dmu_dE = 1.0 / (2.0 * (1.0 + nu))
        dla_dE = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        
        # 执行投影！
        grad_E = grad_mu * dmu_dE + grad_la * dla_dE
        
        # =====================================================
        
        # 5. 归一化与 Hessian 预条件 (复用父类逻辑)
        grad_s = grad_E * self.E_scaling
        grad_conditioned = grad_s * self.preconditioner
        
        return loss, grad_conditioned
    
    def solve(self, E_guess_physical=None, total_steps=150):
        """
        重写 solve 方法：放宽了参数 s 的截断范围 (Clipping Range)，
        防止在高刚度基体 (s_base > 20) 的情况下优化器卡死。
        """
        # 1. 全局背景校准 (复用父类方法)
        # 这会寻找一个最佳的均匀标量 s_base，使 Loss 最小
        s_base_val = self.fit_background_stiffness()
        
        print(f"=== 阶段二：周期性回撤反演 (Cyclic Erosion) [Tensor Mode] ===")
        
        # 初始化 s 向量
        s = np.ones(self.num_cells) * s_base_val
        s_base = np.ones_like(s) * s_base_val
        
        # [关键修改] 动态设定截断范围
        # 如果 s_base 很大 (例如 100)，我们需要允许 s 变得更大
        # 设定下限为 0.01 (防止除零或负数)
        # 设定上限为 s_base 的 10 倍或 500.0 (取大者)
        clip_min = 0.01
        clip_max = max(500.0, s_base_val * 10.0)
        
        print(f"  [Config] Dynamic Clipping Range: [{clip_min:.2f}, {clip_max:.2f}]")
        print(f"           (Based on s_base={s_base_val:.2f})")
        
        # 优化超参数
        cycle_length = 20    # 周期长度
        erosion_rate = 0.7   # 回撤保留率
        learning_rate = 2.0  # 学习率
        
        for k in range(total_steps):
            
            # --- 1. 周期性回撤 (The Reset) ---
            if k > 0 and k % cycle_length == 0:
                print(f"  >>> [Cycle Reset] 触发回撤：保留 {100*erosion_rate:.0f}% 更新量，其余回归基体 <<<")
                s = s_base + (s - s_base) * erosion_rate
            
            # --- 2. 计算梯度 ---
            # 调用重写过的 _objective_and_gradient (基于张量基底)
            loss, grad = self._objective_and_gradient(s)
            
            # --- 3. 梯度归一化与更新 ---
            grad_mean = np.mean(np.abs(grad))
            
            # 防止梯度过小导致除零
            if grad_mean < 1e-20:
                print("  [Converged] Gradient too small.")
                break
                
            # 梯度截断 (防止单个单元梯度爆炸)
            grad = np.clip(grad, -10*grad_mean, 10*grad_mean)
            
            # 归一化方向 (Unit Vector Step)
            grad_norm = np.linalg.norm(grad)
            update = - learning_rate * (grad / grad_norm)
            
            s_new = s + update
            
            # --- 4. [关键] 应用放宽后的物理约束 ---
            s_new = np.clip(s_new, clip_min, clip_max)
            
            # --- 5. 统计与保存 ---
            change = np.linalg.norm(s_new - s)
            current_mean_E = np.mean(s_new * self.E_scaling)
            
            print(f"  [Step {k}] Loss={loss:.2e} | Mean E={current_mean_E:.0f} | Mag={change:.4f}")
            
            s = s_new
            
            # 定期保存 VTK
            if k % 5 == 0:
                self._export_intermediate_vtk(s * self.E_scaling, step=k)
            
        return s * self.E_scaling