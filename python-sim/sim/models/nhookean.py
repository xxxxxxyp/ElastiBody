import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os
import sys

class NHookeanForwardSolver:
    def __init__(self, initializer):
        """
        初始化前向求解器
        :param initializer: 已经初始化好的 ElasticInitializer 实例 (持有 C++ backend)
        """
        self.init = initializer
        self.cpp = initializer.cpp_backend
        self.num_nodes = initializer.num_nodes
        
        # 状态变量
        self.u = np.zeros(self.num_nodes * 3) # 当前位移向量 (flattened)
        self.f_ext = np.zeros(self.num_nodes * 3) # 外力向量

        self.num_dofs = self.num_nodes * 3
        
        # 边界条件缓存
        self.fixed_dofs = [] # 被固定的自由度索引列表
        
    def set_dirichlet_bc(self, fixed_nodes_indices):
        """
        设置 Dirichlet 边界条件 (全约束: x, y, z 都固定)
        :param fixed_nodes_indices: 节点索引列表
        """
        dofs = []
        for n_idx in fixed_nodes_indices:
            dofs.extend([3*n_idx, 3*n_idx+1, 3*n_idx+2])
        self.fixed_dofs = np.array(dofs, dtype=int)
        print(f"[Solver] Dirichlet BCs set: {len(fixed_nodes_indices)} nodes fixed ({len(self.fixed_dofs)} DOFs).")

    def _apply_bc_to_system(self, K, R):
        """
        对线性系统 K * du = R 应用边界条件
        方法：将固定自由度对应的 K 行/列置零，对角线置 1；R 置 0。
        这强制使得 du = 0。
        """
        if len(self.fixed_dofs) == 0:
            return K, R
            
        # 转换为 LIL 格式以便高效修改结构 (虽然慢一点，但比 CSR 修改安全)
        # 考虑到性能，也可以在 CSR 上利用切片掩膜操作，这里为了稳健性使用覆盖法
        
        # 策略：不修改稀疏矩阵结构，而是利用数学技巧
        # 对于很大的矩阵，修改结构很慢。
        # 这里使用一种常见技巧：Penalty Method 或者将行/列归零
        
        # 1. 处理残差 R: 固定点的残差设为 0
        R[self.fixed_dofs] = 0.0
        
        # 2. 处理矩阵 K: 
        # 这是一个计算密集型操作。为了 Python 端的性能，我们修改对角线
        # 将固定行的非对角元素设为0太慢了。
        # 替代方案：将对角线元素设为一个巨大的数 (Penalty Method)
        # K_ii = 1e15 * max(diag(K))
        # 这样求解出来的 du_i 就会趋近于 0
        
        penalty = 1e18 # 足够大的数
        
        # K 是 scipy.sparse.csc_matrix 或 csr_matrix
        # 我们可以直接修改 data 数组吗？比较危险。
        # 让我们构建一个对角修正矩阵
        
        # 更标准的方法（虽然繁琐）：
        # 将 K 转换为 LIL，归零行和列，对角置1。
        # 但在 Python 中这可能需要几秒钟。
        
        # === 高性能 Penalty 方案 ===
        # 在对角线上叠加巨大数值
        diag_indices = self.fixed_dofs
        
        # 构建一个稀疏矩阵，只在 fixed_dofs 的对角线上有值
        data = np.ones(len(diag_indices)) * penalty
        rows = diag_indices
        cols = diag_indices
        K_penalty = sp.coo_matrix((data, (rows, cols)), shape=K.shape)
        
        K_mod = K + K_penalty
        return K_mod, R

    def solve_static_step(self, force_input, step_index=0, tol=1e-4, max_iter=10):
        """
        :param force_input: 
            可以是 float (标量): 会被自动施加到所有 Top Nodes (Z方向)
            可以是 np.array (向量): 形状 (3*N,) 或 (N, 3)，直接作为外力向量 f_ext
        """
        # 1. 初始化外力向量
        self.f_ext = np.zeros(self.num_dofs)
        
        # 2. 判断输入类型
        if isinstance(force_input, (float, int)):
            # === 旧逻辑：标量均摊 ===
            # 如果还没找过受力节点，找一下
            if not hasattr(self, 'force_application_nodes'):
                 z_coords = self.init.nodes[:, 2]
                 z_max = np.max(z_coords)
                 self.force_application_nodes = np.where(z_coords > z_max - 1e-5)[0]
            
            # 施加 Z 向力
            for node_idx in self.force_application_nodes:
                self.f_ext[3*node_idx + 2] = force_input
                
        elif isinstance(force_input, np.ndarray):
            # === 新逻辑：直接映射 (您想要的) ===
            # 自动展平为 (3N,)
            f_flat = force_input.flatten()
            
            if f_flat.shape[0] != self.num_dofs:
                raise ValueError(f"Force vector shape mismatch! Expected {self.num_dofs}, got {f_flat.shape[0]}")
            
            self.f_ext = f_flat
            
        else:
            raise TypeError("force_input must be float or numpy array")
            
        # Newton-Raphson 迭代
        for k in range(max_iter):
            # 1. 注入位移
            self.cpp.set_current_displacement(self.u)
                
            # 2. 获取 K 和 f_int
            K_tangent = self.cpp.gen_tangent_stiffness() 
            f_int = np.array(self.cpp.gen_grad_f())
                
            # 3. 计算残差 [CRITICAL FIX]
            # 依据 C++ ElasticBody::eqn 的逻辑: dP = -(f_sensor + f_inter)
            # 这里的 f_ext 对应 f_sensor, f_int 对应 f_inter
            residual = - (self.f_ext + f_int)
                
            # [新增] 安全检查
            if np.any(np.isnan(residual)):
                raise RuntimeError("Solver exploded: Residual contains NaN.")

            # 4. 应用边界条件
            K_mod, residual_mod = self._apply_bc_to_system(K_tangent, residual)
                
            # 5. 检查收敛
            res_norm = np.linalg.norm(residual_mod)
            print(f"  Iter {k}: Residual Norm = {res_norm:.4e}")
                
            if res_norm < tol:
                print("  -> Converged.")
                break
                
            # 6. 求解
            try:
                du = spla.spsolve(K_mod, residual_mod)
            except Exception as e:
                print(f"  [Error] Linear solver failed: {e}")
                break
                    
            # 7. 更新位移
            self.u += du
                
        self._export_data(step_index)

    def _export_data(self, step_idx):
        """
        导出符合 ElasticBody::load_data 规范的数据
        包括 pnt{i}.txt, force{i}.txt, A-{i}.txt, B-{i}.txt
        """
        base_dir = self.init.output_dir
        
        # 1. pnt{i}.txt: 变形后的节点坐标 (flattened)
        # nodes_current = nodes0 + u
        current_nodes = self.init.nodes.flatten() + self.u
        np.savetxt(os.path.join(base_dir, f"pnt{step_idx}.txt"), current_nodes)
        
        # 2. force{i}.txt: 外力
        np.savetxt(os.path.join(base_dir, f"force{step_idx}.txt"), self.f_ext)
        
        # 3. A-{i}.txt 和 B-{i}.txt (稀疏矩阵)
        # 这里的 A 和 B 是为了反向优化准备的。
        # 前向生成的数据通常假设所有点都可见 (B=Identity) 且没有额外正则项 (A=Zero)
        
        # 生成单位矩阵 B (Size: 3N x 3N)
        # 为了节省空间，我们只写入对角线非零值
        # 格式: row col value
        size = 3 * self.num_nodes
        indices = np.arange(size)
        ones = np.ones(size)
        
        # B = Identity
        B_data = np.column_stack((indices, indices, ones))
        self._write_sparse_txt(os.path.join(base_dir, f"B-{step_idx}.txt"), B_data)
        
        # A = Zero (或者空文件，视读取器鲁棒性而定，这里写一个空的)
        # 为了安全起见，写入一个全零行或者直接创建一个空文件
        # example.py 中 A 是用来做 regularization 或 boundary penalty 的
        # 我们写入一个 dummy entry: 0 0 0
        A_data = np.array([[0, 0, 0]]) 
        self._write_sparse_txt(os.path.join(base_dir, f"A-{step_idx}.txt"), A_data)
        
        print(f"  [IO] Exported step {step_idx} data to {base_dir}")

    def _write_sparse_txt(self, filename, data_array):
        # 格式化输出: int int float
        # data_array shape: (N, 3)
        fmt = "%d %d %.6f"
        np.savetxt(filename, data_array, fmt=fmt)