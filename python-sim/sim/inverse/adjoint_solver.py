import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os
import time

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

        # [核心] 计算照明图 (Illumination Map) / 伪 Hessian 对角线
        # 这是一次性的物理计算，用来衡量每个单元对测量结果的贡献度
        print("[Init] 正在计算 Hessian 预条件因子 (这代表纯物理的灵敏度分布)...")
        self.preconditioner = self._compute_hessian_diag()

    def _compute_hessian_diag(self):
        # 我们需要计算 J^T * J 的对角线
        # J (Sensitivity Matrix) 的每一列代表：该单元变硬一点，所有节点位移变多少？
        
        # 1. 获取灵敏度矩阵 S (3N x M)
        # 注意：这步比较耗时，但为了物理严谨性必须算
        S = self.sens_builder.build_sensitivity_matrix()
        
        # 2. 计算每一列的模长平方 (Column Norms)
        # diag_H[i] = sum(S[:, i]**2)
        # 这代表第 i 个单元的“总灵敏度能量”
        # S 是 scipy.sparse.csc_matrix 或 csr_matrix
        # 我们可以利用矩阵乘法快速计算: sum(S_ij * S_ij)
        
        # 如果 S 很大，直接点乘可能会爆内存。我们可以手动算列模长。
        # 这里假设 S 是 CSC 格式 (列压缩)，算列模长很快
        S_csc = S.tocsc()
        diag_H = np.zeros(S.shape[1])
        
        for i in range(S.shape[1]):
            col = S_csc.getcol(i)
            diag_H[i] = np.sum(col.data**2)
            
        # 3. 归一化处理
        # 避免除以 0，加一个小的阻尼
        max_val = np.max(diag_H)
        diag_H = diag_H + 1e-6 * max_val
        
        # 4. 计算预条件因子 P = 1 / sqrt(diag_H) 
        # (有些文献用 1/H，有些用 1/sqrt(H) 作为梯度缩放，后者更稳健)
        preconditioner = 1.0 / np.sqrt(diag_H)
        
        # 归一化让它均值为 1，方便调参
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
            # 伴随方程 K.T * lambda = -residual
            lam = spla.spsolve(K_mod.T, -residual_bc) 
        except:
            lam = np.zeros_like(residual)
        return lam

    def _objective_and_gradient(self, s_vec):
        # 1. Forward
        E_physical = s_vec * self.E_scaling
        try:
            u_sim = self._solve_forward(E_physical)
        except:
            return 1e9, np.zeros_like(s_vec)
        
        # 2. Adjoint
        diff = u_sim - self.u_obs
        diff[self.fixed_dofs] = 0.0 
        loss = 0.5 * np.sum(diff**2)
        lam = self._solve_adjoint(u_sim, diff)
        
        # 3. Raw Gradient (这是纯物理梯度)
        S = self.sens_builder.build_sensitivity_matrix()
        grad_E = S.T @ lam
        grad_s = grad_E * self.E_scaling
        
        # === [核心] 纯物理预条件 (Hessian Preconditioning) ===
        # 我们不改变形状，我们只是根据物理灵敏度进行缩放。
        # 灵敏度低的地方（深处），Hessian小，梯度被放大。
        # 灵敏度高的地方（表面），Hessian大，梯度被缩小。
        # 这是牛顿法的本质，不是人工修饰。
        
        grad_conditioned = grad_s * self.preconditioner
        
        # ===================================================
        
        return loss, grad_conditioned

    def solve(self, E_guess_physical, max_iter=50):
        print(f"=== 启动 Hessian 预条件物理反演 ===")
        
        s = E_guess_physical / self.E_scaling
        
        # 学习率
        learning_rate = 2.0 
        
        for k in range(max_iter):
            loss, grad = self._objective_and_gradient(s)
            
            # 梯度截断，防止奇异点
            grad_mean = np.mean(np.abs(grad))
            grad = np.clip(grad, -10*grad_mean, 10*grad_mean)
            
            # 归一化方向 (Trust Region 思想)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-20: break
            
            # 更新：负梯度方向 (变硬)
            update = - learning_rate * (grad / grad_norm)
            
            s_new = s + update
            
            # 物理约束：不让 E 变成负数
            s_new = np.clip(s_new, 0.2, 20.0) # 100k ~ 10MPa
            
            change = np.linalg.norm(s_new - s)
            current_mean = np.mean(s_new * self.E_scaling)
            
            print(f"  [Step {k}] Loss={loss:.2e} | Mean E={current_mean:.0f} | Update Mag={change:.4f}")
            
            s = s_new
            self._export_intermediate_vtk(s * self.E_scaling)
            
        return s * self.E_scaling

    def _export_intermediate_vtk(self, E_vec):
        import meshio
        path = os.path.join(self.init.output_dir, "adjoint_current.vtk")
        mesh = meshio.Mesh(
            points=self.init.nodes,
            cells=[("tetra", self.init.cells)],
            cell_data={"E_Recon": [E_vec]}
        )
        mesh.write(path)