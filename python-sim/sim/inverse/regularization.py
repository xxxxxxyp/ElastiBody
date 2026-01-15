import numpy as np
import scipy.sparse as sp

class Regularizer:
    def __init__(self, cells):
        self.cells = cells
        self.num_cells = len(cells)
        self.L = None

    def build_laplacian(self):
        """
        构建拓扑拉普拉斯矩阵 L (N_cells x N_cells)
        用于平滑约束: || L * E ||^2
        """
        if self.L is not None:
            return self.L

        print("[Regularization] Building Laplacian matrix...")
        # 1. 构建邻接图 (Adjacency Graph)
        # 策略: 两个单元如果共享一个面(3个节点)，则是邻居
        
        # 提取所有面
        # 每个四面体有4个面，节点组合为 (0,1,2), (0,1,3), (0,2,3), (1,2,3)
        faces = np.vstack([
            self.cells[:, [0, 1, 2]],
            self.cells[:, [0, 1, 3]],
            self.cells[:, [0, 2, 3]],
            self.cells[:, [1, 2, 3]]
        ])
        
        # 对每个面的节点索引排序，以便去重
        faces.sort(axis=1)
        
        # 记录每个面属于哪个单元
        # 使用 numpy 的 unique 和 inverse 索引
        # 这种方法比 Python 字典循环快得多
        
        # 给每个面赋予一个唯一的哈希值或结构化视图
        # 既然节点索引是整数，我们可以通过结构化数组处理
        dtype = [('n1', int), ('n2', int), ('n3', int)]
        faces_struct = np.array([tuple(f) for f in faces], dtype=dtype)
        
        # 排序并找到重复的面
        # argsort
        sorted_indices = np.argsort(faces_struct, order=('n1', 'n2', 'n3'))
        sorted_faces = faces_struct[sorted_indices]
        
        # 查找相邻的相同面
        # 如果 sorted_faces[i] == sorted_faces[i+1]，说明这两个面是同一个几何面
        # 对应的原始索引 sorted_indices[i] 和 sorted_indices[i+1] 属于两个不同的单元
        
        # 原始 faces 数组中，索引 k 对应的单元索引是 k % num_cells
        # 因为我们是 vstack 了 4 块
        
        adj_pairs = []
        
        N = len(faces)
        # 遍历排序后的面，找相同的对
        for i in range(N - 1):
            if sorted_faces[i] == sorted_faces[i+1]:
                # 找到共享面
                idx1 = sorted_indices[i]
                idx2 = sorted_indices[i+1]
                
                # 映射回单元索引
                cell1 = idx1 % self.num_cells
                cell2 = idx2 % self.num_cells
                
                if cell1 != cell2:
                    adj_pairs.append((cell1, cell2))
        
        # 2. 构建 L 矩阵
        # L_ii = degree, L_ij = -1
        data = []
        rows = []
        cols = []
        degrees = np.zeros(self.num_cells)
        
        for c1, c2 in adj_pairs:
            # L_ij = -1
            rows.append(c1); cols.append(c2); data.append(-1.0)
            rows.append(c2); cols.append(c1); data.append(-1.0)
            degrees[c1] += 1
            degrees[c2] += 1
            
        # 对角线
        rows.extend(range(self.num_cells))
        cols.extend(range(self.num_cells))
        data.extend(degrees)
        
        self.L = sp.csc_matrix((data, (rows, cols)), shape=(self.num_cells, self.num_cells))
        print(f"[Regularization] Laplacian built. {len(adj_pairs)} neighbor pairs found.")
        return self.L