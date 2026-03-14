import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import os
from .sensitivity import SensitivityBuilder
from .regularization import Regularizer
from ..models.nhookean import NHookeanForwardSolver

SENSITIVITY_EPSILON_SCALE = 1e-10
PHYSICAL_MIN_E = 2000.0
PHYSICAL_MAX_E = 500000.0

class InverseSolver:
    def __init__(self, initializer, obs_step_idx=9, obs_step_list=None):
        self.init = initializer
        self.cpp = initializer.cpp_backend
        self.model_name = initializer.model_name
        self.obs_step_list = self._normalize_obs_step_list(obs_step_idx, obs_step_list)
        self.step_idx = self.obs_step_list[0]
        
        # 加载观测数据
        data_dir = os.path.join(initializer.output_dir) # Use init's output dir
        if not os.path.exists(data_dir):
            # Fallback
            data_dir = f"data-sampling/{self.model_name}"

        self.nodes_obs_list = []
        self.u_meas_list = []
        self.f_ext_list = []

        nodes_0 = self.init.nodes.flatten()
        for step_idx in self.obs_step_list:
            pnt_path = os.path.join(data_dir, f"pnt{step_idx}.txt")
            force_path = os.path.join(data_dir, f"force{step_idx}.txt")

            if not os.path.exists(pnt_path):
                raise FileNotFoundError(f"Observation data {pnt_path} not found.")
            if not os.path.exists(force_path):
                raise FileNotFoundError(f"Force data {force_path} not found.")

            nodes_obs = np.loadtxt(pnt_path).reshape(-1)
            f_ext = np.loadtxt(force_path).reshape(-1)

            if len(nodes_obs) != len(nodes_0):
                raise ValueError(
                    f"Observation step {step_idx} has {len(nodes_obs)} values, expected {len(nodes_0)}."
                )
            if len(f_ext) != len(nodes_0):
                raise ValueError(
                    f"Force step {step_idx} has {len(f_ext)} values, expected {len(nodes_0)}."
                )

            self.nodes_obs_list.append(nodes_obs)
            self.u_meas_list.append(nodes_obs - nodes_0)
            self.f_ext_list.append(f_ext)

        self.num_states = len(self.obs_step_list)
        # 兼容旧版单工况调用
        self.nodes_obs = self.nodes_obs_list[0]
        self.u_meas_full = self.u_meas_list[0]
        self.f_ext = self.f_ext_list[0]
        
        # 表面节点用于填补步骤的 Dirichlet BC
        self.surface_indices = self._find_boundary_nodes()
        
        self.sensitivity_builder = SensitivityBuilder(initializer)
        self.regularizer = Regularizer(initializer.cells)
        self.L_matrix = self.regularizer.build_laplacian()
        self.current_E = np.full(initializer.num_cells, initializer.E_base)

    def _normalize_obs_step_list(self, obs_step_idx, obs_step_list):
        if obs_step_list is None:
            obs_step_list = [obs_step_idx]
        elif np.isscalar(obs_step_list):
            obs_step_list = [int(obs_step_list)]
        else:
            obs_step_list = list(obs_step_list)

        if len(obs_step_list) == 0:
            raise ValueError("obs_step_list cannot be empty.")
        return [int(step_idx) for step_idx in obs_step_list]

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

    def _normalize_ignore_nodes_list(self, ignore_nodes, ignore_nodes_list):
        if ignore_nodes_list is None:
            if ignore_nodes is None:
                ignore_nodes_list = [None] * self.num_states
            else:
                ignore_nodes_array = np.asarray(ignore_nodes, dtype=int).reshape(-1)
                ignore_nodes_list = [ignore_nodes_array.copy() for _ in range(self.num_states)]
        else:
            ignore_nodes_entries = list(ignore_nodes_list)
            if self.num_states == 1 and (
                len(ignore_nodes_entries) == 0 or np.isscalar(ignore_nodes_entries[0])
            ):
                ignore_nodes_list = [np.asarray(ignore_nodes_entries, dtype=int).reshape(-1)]
            elif self.num_states == 1 and isinstance(ignore_nodes_list, np.ndarray):
                ignore_nodes_list = [np.asarray(ignore_nodes_list, dtype=int).reshape(-1)]
            else:
                ignore_nodes_list = ignore_nodes_entries

        if len(ignore_nodes_list) != self.num_states:
            raise ValueError(
                f"ignore_nodes_list length ({len(ignore_nodes_list)}) must match obs_step_list "
                f"length ({self.num_states})."
            )

        normalized = []
        for nodes in ignore_nodes_list:
            if nodes is None:
                normalized.append(None)
            else:
                normalized.append(np.asarray(nodes, dtype=int).reshape(-1))
        return normalized

    def _normalize_observed_nodes_list(self, observed_nodes_list):
        if observed_nodes_list is None:
            observed_nodes_list = [None] * self.num_states
        else:
            observed_nodes_entries = list(observed_nodes_list)
            if self.num_states == 1 and (
                len(observed_nodes_entries) == 0 or np.isscalar(observed_nodes_entries[0])
            ):
                observed_nodes_list = [np.asarray(observed_nodes_entries, dtype=int).reshape(-1)]
            elif self.num_states == 1 and isinstance(observed_nodes_list, np.ndarray):
                observed_nodes_list = [np.asarray(observed_nodes_list, dtype=int).reshape(-1)]
            else:
                observed_nodes_list = observed_nodes_entries

        if len(observed_nodes_list) != self.num_states:
            raise ValueError(
                f"observed_nodes_list length ({len(observed_nodes_list)}) must match obs_step_list "
                f"length ({self.num_states})."
            )

        normalized = []
        for nodes in observed_nodes_list:
            if nodes is None:
                normalized.append(None)
            else:
                normalized.append(np.asarray(nodes, dtype=int).reshape(-1))
        return normalized

    def _build_valid_dof_masks(self, ignore_nodes_list):
        valid_dof_masks = []
        bc_nodes_list = []
        for ignore_nodes in ignore_nodes_list:
            valid_dof_mask = np.ones(3 * self.init.num_nodes, dtype=bool)
            if ignore_nodes is None or len(ignore_nodes) == 0:
                bc_nodes = self.surface_indices
            else:
                bc_nodes = ignore_nodes
                ignored_dofs = np.concatenate(
                    [np.arange(3 * node_idx, 3 * node_idx + 3) for node_idx in ignore_nodes]
                )
                valid_dof_mask[ignored_dofs] = False

            valid_dof_masks.append(valid_dof_mask)
            bc_nodes_list.append(np.asarray(bc_nodes, dtype=int))
        return valid_dof_masks, bc_nodes_list

    def _compute_sensitivity_weights(self, S_valid, alpha):
        sensitivity_magnitude = np.asarray(sp.linalg.norm(S_valid, axis=0)).ravel()
        max_magnitude = np.max(sensitivity_magnitude) if sensitivity_magnitude.size > 0 else 0.0
        if max_magnitude <= 0.0:
            return np.ones(self.init.num_cells)

        # 用当前工况灵敏度最大值的极小比例作为稳定项，避免零列或超弱响应时分母为零。
        epsilon = SENSITIVITY_EPSILON_SCALE * max_magnitude
        raw_weights = 1.0 / (sensitivity_magnitude**alpha + epsilon)
        return raw_weights / np.mean(raw_weights)

    def _compute_state_weight(self, u_state, f_ext_masked):
        force_norm = np.linalg.norm(f_ext_masked)
        disp_norm = np.linalg.norm(u_state)
        reference_norm = max(force_norm, disp_norm, 1.0)
        return 1.0 / reference_norm

    def solve_alternating(
        self,
        lambda_reg=1e-6,
        max_iter=10,
        ignore_nodes=None,
        ignore_nodes_list=None,
        observed_nodes_list=None,
        alpha=0.5,
    ):
        """
        :param lambda_reg: 正则化系数 (建议极小, e.g. 1e-12 ~ 1e-14)
        :param alpha: 灵敏度加权指数 (0.0=无加权, 0.5=适度, 1.0=强)
        """
        print(f"\n=== Starting Inverse Optimization (Reg={lambda_reg}, Alpha={alpha}) ===")

        ignore_nodes_list = self._normalize_ignore_nodes_list(ignore_nodes, ignore_nodes_list)
        observed_nodes_list = self._normalize_observed_nodes_list(observed_nodes_list)
        valid_dof_masks, bc_nodes_list = self._build_valid_dof_masks(ignore_nodes_list)

        for state_idx, ignore_nodes_state in enumerate(ignore_nodes_list):
            if ignore_nodes_state is not None and len(ignore_nodes_state) > 0:
                print(
                    f"[Inverse] State {state_idx}: excluding {len(ignore_nodes_state)} fixed nodes "
                    "from force balance."
                )

        for k in range(max_iter):
            print(f"\n--- Iteration {k} ---")
            
            self.init.E_field = self.current_E
            self.init.commit_to_cpp()

            A_sys_total = sp.csc_matrix((self.init.num_cells, self.init.num_cells))
            b_sys_total = np.zeros(self.init.num_cells)
            state_weight_vectors = []

            # ==========================================================
            # Joint Inversion 核心：
            # 逐个工况做“前向求解 -> 灵敏度构建 -> 加权”，然后把每个工况
            # 的信息矩阵 S_i^T S_i 和右端项 S_i^T (-f_i) 累加起来。
            # 这样多时刻/多方向的观测会共同约束同一个 E_field。
            # ==========================================================
            for state_idx in range(self.num_states):
                u_meas_state = self.u_meas_list[state_idx]
                f_ext_state = self.f_ext_list[state_idx]
                valid_dof_mask = valid_dof_masks[state_idx]
                f_ext_masked = f_ext_state[valid_dof_mask]

                fwd_solver = NHookeanForwardSolver(self.init)
                fixed_nodes = (
                    bc_nodes_list[state_idx]
                    if (bc_nodes_list is not None and bc_nodes_list[state_idx] is not None)
                    else np.array([], dtype=int)
                )
                observed_nodes = (
                    observed_nodes_list[state_idx]
                    if (observed_nodes_list is not None and observed_nodes_list[state_idx] is not None)
                    else np.array([], dtype=int)
                )
                fwd_bc_nodes = np.unique(np.concatenate((fixed_nodes, observed_nodes)))

                if len(fwd_bc_nodes) > 0:
                    fwd_solver.set_dirichlet_bc(fwd_bc_nodes)
                else:
                    fwd_solver.set_dirichlet_bc([])
                fwd_solver.u = u_meas_state.copy()

                fwd_solver.solve_static_step(f_ext_state, step_index=self.obs_step_list[state_idx], tol=1e-5)
                u_full_state = fwd_solver.u

                self.cpp.set_current_displacement(u_full_state)
                S_full = self.sensitivity_builder.build_sensitivity_matrix()
                S_valid = S_full[valid_dof_mask, :]

                cell_weights = self._compute_sensitivity_weights(S_valid, alpha)
                W_diag = sp.diags(cell_weights)
                state_weight_vectors.append(cell_weights)

                # ------------------------------------------------------
                # 数据权重平衡：
                # 大载荷/大位移工况会天然产生更大的响应，如果不做归一化，
                # 它们会在联合反演中完全主导 A、b 的累加。
                # 这里用“外力范数 / 位移范数”的量级为每个工况生成一个标量权重，
                # 让不同工况贡献处于相近数量级，从而提升深层组织反演的稳定性。
                # ------------------------------------------------------
                state_weight = self._compute_state_weight(u_full_state[valid_dof_mask], f_ext_masked)
                S_weighted = state_weight * (S_valid @ W_diag)

                A_sys_total = A_sys_total + (S_weighted.T @ S_weighted)
                b_sys_total = b_sys_total + (S_weighted.T @ (-f_ext_masked))

            mean_weights = np.mean(np.vstack(state_weight_vectors), axis=0)
            W_reg = sp.diags(mean_weights)

            # ----------------------------------------------------------
            # 正则化统一在所有工况累加完成后再加入：
            # 这样 Reg 只作为“共同的材料场先验”，不会重复计入某个单一工况。
            # 我们用各工况单元权重的平均值来缩放拉普拉斯项，兼顾多工况灵敏度分布。
            # ----------------------------------------------------------
            Reg = lambda_reg * (self.L_matrix @ W_reg).T @ (self.L_matrix @ W_reg)
            A_sys = A_sys_total + Reg

            # ----------------------------------------------------------
            # 将原线性方程 A_sys x = b_sys 转写为等价的二次型最小化问题：
            #
            #   f(x) = 1/2 x^T A_sys x - b_sys^T x
            #
            # 对该目标函数求梯度可得：
            #
            #   ∇f(x) = A_sys x - b_sys
            #
            # 这样做的好处是：我们可以在求解 E_tilde 时直接引入物理边界约束，
            # 避免无约束线性求解在低灵敏度区域被噪声放大，产生极高/极低的振荡伪影。
            # 下面统一使用稀疏矩阵 .dot() 来计算矩阵向量乘法，兼顾数值效率与内存占用。
            # ----------------------------------------------------------
            max_diag = np.max(np.abs(A_sys.diagonal()))
            scale_factor = 1.0 / max_diag if max_diag > 1e-20 else 1.0

            A_opt = A_sys * scale_factor
            b_opt = b_sys_total * scale_factor

            def objective_fn(x):
                return 0.5 * x.T @ A_opt.dot(x) - b_opt.dot(x)

            def gradient_fn(x):
                return A_opt.dot(x) - b_opt

            # ----------------------------------------------------------
            # 最终材料场满足 E_new = mean_weights * E_tilde，因此优化变量 E_tilde
            # 的边界必须按 mean_weights 反向缩放。
            #
            # 对每个单元 i：
            #   PHYSICAL_MIN_E <= mean_weights[i] * E_tilde[i] <= PHYSICAL_MAX_E
            #
            # 等价得到：
            #   PHYSICAL_MIN_E / mean_weights[i] <= E_tilde[i]
            #       <= PHYSICAL_MAX_E / mean_weights[i]
            #
            # 这保证优化器求出的 E_new 天然落在物理可接受区间内，不再需要事后
            # 的符号翻转或硬截断。
            # ----------------------------------------------------------
            bounds = [
                (
                    PHYSICAL_MIN_E / mean_weights[i],
                    PHYSICAL_MAX_E / mean_weights[i],
                )
                for i in range(self.init.num_cells)
            ]
            x0 = self.current_E / mean_weights

            try:
                res = opt.minimize(
                    objective_fn,
                    x0,
                    jac=gradient_fn,
                    bounds=bounds,
                    method="L-BFGS-B",
                    options={"ftol": 1e-9, "gtol": 1e-5},
                )
            except Exception as e:
                print(f"[Error] Optimization solve failed: {e}")
                break

            if not res.success:
                print(f"[Error] Optimization did not converge: {res.message}")
                break

            E_tilde = res.x
            
            # Step 4: Recover
            E_new = mean_weights * E_tilde
            
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
