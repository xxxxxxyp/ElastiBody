import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import minimize
import meshio
import os

from ..models.nhookean import NHookeanForwardSolver
from .sensitivity import SensitivityBuilder

class AdjointSolver:
    def __init__(self, initializer, u_obs_full, f_ext, fixed_dofs):
        self.init = initializer
        self.cpp = initializer.cpp_backend
        self.u_obs = u_obs_full
        self.f_ext = f_ext
        self.fixed_dofs = fixed_dofs
        
        self.fwd_solver = NHookeanForwardSolver(initializer)
        self.fwd_solver.fixed_dofs = fixed_dofs 
        self.sens_builder = SensitivityBuilder(initializer)
        
        self.E_scaling = 500000.0 
        self.iteration_count = 0

        # [Hessian] 计算全逆 Hessian 近似 (Full Inverse Hessian)
        # 这是物理上的“深层放大镜”，必须保留
        print("[Init] 正在计算全逆 Hessian 补偿...")
        self.preconditioner = self._compute_hessian_diag()

    def _compute_hessian_diag(self):
        S = self.sens_builder.build_sensitivity_matrix()
        S_csc = S.tocsc()
        diag_H = np.zeros(S.shape[1])
        for i in range(S.shape[1]):
            col = S_csc.getcol(i)
            diag_H[i] = np.sum(col.data**2)
            
        max_H = np.max(diag_H)
        diag_H = diag_H + 1e-9 * max_H
        
        # Power = 1.0 (牛顿法近似)
        preconditioner = 1.0 / diag_H 
        
        # 限制放大倍数，防止数值爆炸
        p_min = np.min(preconditioner)
        p_limit = p_min * 1e6 
        preconditioner = np.clip(preconditioner, p_min, p_limit)
        
        # 归一化
        preconditioner = preconditioner / np.mean(preconditioner)
        print(f"  [Hessian] Surface/Deep Ratio: {np.max(preconditioner)/np.min(preconditioner):.2f}")
        return preconditioner

    def _solve_forward(self, E_vec):
        self.init.E_field = E_vec
        self.init.commit_to_cpp()
        self.cpp.set_element_modulus(E_vec)
        self.fwd_solver.solve_static_step(self.f_ext, step_index=999, tol=1e-4, max_iter=20)
        return self.fwd_solver.u

    def _solve_adjoint(self, u_current, residual):
        self.cpp.set_current_displacement(u_current)
        K_T = self.cpp.gen_tangent_stiffness()
        residual_bc = residual.copy()
        residual_bc[self.fixed_dofs] = 0.0
        K_mod, _ = self.fwd_solver._apply_bc_to_system(K_T, residual_bc)
        try:
            lam = spla.spsolve(K_mod.T, -residual_bc) 
        except:
            lam = np.zeros_like(residual)
        return lam

    def _objective_and_gradient(self, s_vec):
        E_physical = s_vec * self.E_scaling
        try:
            u_sim = self._solve_forward(E_physical)
        except:
            return 1e9, np.zeros_like(s_vec)
        
        diff = u_sim - self.u_obs
        diff[self.fixed_dofs] = 0.0 
        loss = 0.5 * np.sum(diff**2)
        lam = self._solve_adjoint(u_sim, diff)
        
        S = self.sens_builder.build_sensitivity_matrix()
        grad_E = S.T @ lam
        grad_s = grad_E * self.E_scaling
        
        # Hessian 补偿
        grad_conditioned = grad_s * self.preconditioner
        
        return loss, grad_conditioned

    # ==========================================
    # 阶段一：全局硬度搜索
    # ==========================================
    def _objective_global_scalar(self, s_scalar):
        val = s_scalar[0]
        E_physical = np.ones(len(self.init.cells)) * val * self.E_scaling
        try:
            u_sim = self._solve_forward(E_physical)
        except:
            return 1e9
        diff = u_sim - self.u_obs
        diff[self.fixed_dofs] = 0.0
        return 0.5 * np.sum(diff**2)

    def fit_background_stiffness(self):
        print("=== 阶段一：全局背景校准 ===")
        s0 = [1.0] 
        bounds = [(0.01, 100.0)]
        res = minimize(self._objective_global_scalar, s0, method='Nelder-Mead', bounds=bounds, tol=1e-3)
        print(f"=== 背景校准完成: s_base={res.x[0]:.4f} ===")
        return res.x[0]

    # ==========================================
    # 阶段二：周期性回撤反演 (Cyclic Erosion)
    # ==========================================
    def solve(self, E_guess_physical=None, total_steps=150):
        # 1. 先找准基体
        s_base_val = self.fit_background_stiffness()
        
        print(f"=== 阶段二：周期性回撤反演 (Cyclic Erosion) ===")
        print("  [策略] 每隔N步强制 '遗忘' 部分更新，打破局部最优硬壳")
        
        s = np.ones(len(self.init.cells)) * s_base_val
        s_base = np.ones_like(s) * s_base_val
        
        # 参数设置
        cycle_length = 20    # 每 20 步进行一次“大清洗”
        erosion_rate = 0.7   # 清洗时，保留 70% 的当前特征，30% 强制退回基体
                             # 这是一个比较温和的撤销，既不完全重置，又足够破坏硬壳
        
        learning_rate = 2.0
        
        for k in range(total_steps):
            
            # --- [核心机制] 周期性回撤 (The Reset) ---
            if k > 0 and k % cycle_length == 0:
                print(f"  >>> [Cycle Reset] 触发回撤机制：强制衰减 {100*(1-erosion_rate):.0f}% 的更新量 <<<")
                print(f"      (帮助模型跳出 '硬壳' 陷阱，给深层单元重新表现的机会)")
                
                # 公式：s_new = s_base + (s_current - s_base) * rate
                # 作用：把凸出来的硬度强行按回去一部分。
                # 效果：表层硬壳被削弱 -> 残差瞬间变大 -> 梯度重算。
                # 因为有 Hessian 预条件，重算后的梯度会优先分配给深层（性价比更高）。
                s = s_base + (s - s_base) * erosion_rate
                
            
            # --- 常规优化步 ---
            loss, grad = self._objective_and_gradient(s)
            
            # 梯度归一化 (保持步长稳定)
            grad_mean = np.mean(np.abs(grad))
            grad = np.clip(grad, -10*grad_mean, 10*grad_mean)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-20: break
            
            update = - learning_rate * (grad / grad_norm)
            
            s_new = s + update
            
            # 物理约束
            s_new = np.clip(s_new, 0.2, 20.0)
            
            # 统计
            change = np.linalg.norm(s_new - s)
            current_mean = np.mean(s_new * self.E_scaling)
            
            print(f"  [Step {k}] Loss={loss:.2e} | Mean={current_mean:.0f} | Mag={change:.4f}")
            
            s = s_new
            
            # 保存
            if k % 5 == 0:
                self._export_intermediate_vtk(s * self.E_scaling, step=k)
            
        return s * self.E_scaling

    def _export_intermediate_vtk(self, E_vec, step=0):
        # 保存带编号的文件，方便回溯
        path = os.path.join(self.init.output_dir, f"adjoint_step_{step:03d}.vtk")
        mesh = meshio.Mesh(
            points=self.init.nodes,
            cells=[("tetra", self.init.cells)],
            cell_data={"E_Recon": [E_vec]}
        )
        mesh.write(path)