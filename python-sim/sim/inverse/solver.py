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
        data_dir = os.path.join(initializer.output_dir) # Use init's output dir
        if not os.path.exists(data_dir):
            # Fallback
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

    def solve_alternating(self, lambda_reg=1e-12, max_iter=10, ignore_nodes=None, alpha=0.5):
        """
        :param lambda_reg: 正则化系数 (建议极小, e.g. 1e-12 ~ 1e-14)
        :param alpha: 灵敏度加权指数 (0.0=无加权, 0.5=适度, 1.0=强)
        """
        print(f"\n=== Starting Inverse Optimization (Reg={lambda_reg}, Alpha={alpha}) ===")
        
        valid_dof_mask = np.ones(3 * self.init.num_nodes, dtype=bool)
        if ignore_nodes is not None:
            print(f"[Inverse] Excluding {len(ignore_nodes)} fixed nodes from force balance.")
            for idx in ignore_nodes:
                valid_dof_mask[3*idx : 3*idx+3] = False
        
        f_ext_masked = self.f_ext[valid_dof_mask]
        
        PHYSICAL_MIN_E = 2000.0
        
        for k in range(max_iter):
            print(f"\n--- Iteration {k} ---")
            
            # Step 1: Imputation
            # print("1. Predicting internal displacements...")
            self.init.E_field = self.current_E
            self.init.commit_to_cpp()
            
            fwd_solver = NHookeanForwardSolver(self.init)
            fwd_solver.set_dirichlet_bc(self.surface_indices)
            fwd_solver.u = self.u_meas_full.copy() # Set initial guess / BC
            
            # 为了加速，这里可以用较松的容差，或者直接用 u_meas_full 如果是全场观测
            # 但既然我们是表面观测，必须解内部
            fwd_solver.solve_static_step(self.f_ext, step_index=999, tol=1e-5)
            u_full_k = fwd_solver.u
            
            # Step 2: Sensitivity
            # print("2. Building Sensitivity Matrix...")
            self.cpp.set_current_displacement(u_full_k)
            S_full = self.sensitivity_builder.build_sensitivity_matrix()
            S_valid = S_full[valid_dof_mask, :]
            
            # Weighting
            sensitivity_magnitude = sp.linalg.norm(S_valid, axis=0)
            epsilon = 1e-10 * np.max(sensitivity_magnitude)
            
            # Weighting Scheme
            raw_weights = 1.0 / (sensitivity_magnitude**alpha + epsilon)
            weights = raw_weights / np.mean(raw_weights)
            W_diag = sp.diags(weights)
            
            S_weighted = S_valid @ W_diag
            
            # Step 3: Solve
            # print("3. Solving linear system...")
            
            AtA = S_weighted.T @ S_weighted
            Reg = lambda_reg * (self.L_matrix @ W_diag).T @ (self.L_matrix @ W_diag)
            A_sys = AtA + Reg
            
            # [Physics Fix] Force Balance: f_int + f_ext = 0  => S*E = -f_ext
            # 使用 -f_ext_masked 作为右端项
            b_sys = S_weighted.T @ (-f_ext_masked)
            
            try:
                E_tilde = spla.spsolve(A_sys, b_sys)
            except Exception as e:
                print(f"[Error] Linear solve failed: {e}")
                break
            
            # Step 4: Recover
            E_new = weights * E_tilde
            
            # Sign Check (不再应该出现大面积负值)
            neg_frac = np.mean(E_new < 0)
            if neg_frac > 0.5:
                print(f"  [Warn] {neg_frac*100:.1f}% negative E. Physics sign mismatch? Flipping.")
                E_new = -E_new
                
            # Clamp
            E_new = np.maximum(E_new, 2000.0) # 下限 100 Pa
            
            # Stats
            diff = np.linalg.norm(E_new - self.current_E) / np.linalg.norm(self.current_E)
            print(f"  E change: {diff:.4f}")
            print(f"  Stats: Mean={np.mean(E_new):.1f}, Max={np.max(E_new):.1f}, Min={np.min(E_new):.1f}")
            
            self.current_E = E_new
            
            if diff < 0.005:
                print("-> Converged.")
                break
                
        return self.current_E

    def export_result(self, filename="we_recon.txt"):
        path = os.path.join(self.init.output_dir, filename)
        np.savetxt(path, self.current_E)