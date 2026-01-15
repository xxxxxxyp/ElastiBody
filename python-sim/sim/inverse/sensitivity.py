import numpy as np
import scipy.sparse as sp

class SensitivityBuilder:
    def __init__(self, initializer):
        self.cpp = initializer.cpp_backend
        self.cells = initializer.cells # (N_cells, 4)
        self.num_cells = len(self.cells)
        self.num_nodes = initializer.num_nodes
        self.num_dofs = self.num_nodes * 3

    def build_sensitivity_matrix(self):
        """
        构建灵敏度矩阵 S
        :return: scipy.sparse.csc_matrix, shape (3*N_nodes, N_cells)
        """
        # 1. 调用 C++ 获取原始数据 (12 x N_cells)
        # 每一列代表一个单元在 E=1 时产生的 12 个节点力分量
        unit_forces = self.cpp.gen_unit_geometric_forces() 
        
        # 2. 准备稀疏矩阵构建数据 (COO 格式)
        # 我们需要构建 S[global_dof, cell_index] = force_value
        
        # 行索引 (Global DOFs): 需要根据 cells 数组展开
        # cells shape: (N_cells, 4)
        # 我们将其扩展为 (N_cells, 4, 3) 对应的 12 个 DOF
        
        rows = []
        cols = []
        data = []

        # 矢量化构建索引
        # 单元索引 (列索引): 0, 0, ..., 1, 1, ...
        # 每个单元贡献 12 个值，所以列索引重复 12 次
        cell_indices = np.arange(self.num_cells)
        col_indices = np.repeat(cell_indices, 12)
        
        # 节点索引
        # cells_flat: [c0_n0, c0_n1, c0_n2, c0_n3, c1_n0, ...]
        cells_flat = self.cells.flatten() # (4 * N_cells)
        
        # 每个节点有 3 个自由度 (x, y, z)
        # 构造全局 DOF 索引
        # 对每个节点 n，生成 3n, 3n+1, 3n+2
        dof_indices = np.empty((self.num_cells * 4, 3), dtype=int)
        dof_indices[:, 0] = cells_flat * 3
        dof_indices[:, 1] = cells_flat * 3 + 1
        dof_indices[:, 2] = cells_flat * 3 + 2
        
        # 展平为 (12 * N_cells)
        row_indices = dof_indices.flatten() 
        
        # 数据值
        # C++ 返回的是 (12, N_cells)，我们需要按列优先展平（F-order），或者转置后展平
        # 注意: row_indices 的构造顺序是: 
        # Cell 0: Node0_x, Node0_y, Node0_z, Node1_x ...
        # 这正是 unit_forces 每一列的存储顺序
        
        # unit_forces 是 Eigen Matrix (Column Major)，pybind11 转换后通常是 numpy array
        # 我们需要确保展平顺序与 row_indices 一致
        # unit_forces[:, 0] 是 cell 0 的 12 个值
        # 我们需要的数据顺序是: [Cell0_12vals, Cell1_12vals, ...]
        # 所以应该是 unit_forces.T.flatten()
        
        data_values = unit_forces.T.flatten()
        
        # 3. 构建稀疏矩阵
        S = sp.csc_matrix((data_values, (row_indices, col_indices)), 
                          shape=(self.num_dofs, self.num_cells))
        
        return S