import sys
import os
import shutil
import numpy as np
import meshio

try:
    import env_loader
except ImportError:
    pass

import elastic_body_module

class ElasticInitializer:
    def __init__(self, mesh_path, output_dir="data-sampling", E_base=10000.0, nu=0.49):
        """
        前向仿真初始化器 (包含网格方向修正功能)
        """
        self.E_base = E_base
        self.nu = nu
        self.mesh_path = mesh_path
        
        self.model_name = os.path.splitext(os.path.basename(mesh_path))[0]
        self.output_dir = os.path.join(output_dir, self.model_name)
        
        # 1. 准备并修正网格数据
        self.nodes, self.cells = self._load_and_prepare_mesh(mesh_path)
        self.num_nodes = len(self.nodes)
        self.num_cells = len(self.cells)
        
        # 2. 预计算单元几何中心
        self.centroids = self.nodes[self.cells].mean(axis=1)
        
        # 3. 初始化材料场
        self.E_field = np.full(self.num_cells, E_base)
        
        # 4. 实例化 C++ 核心
        print(f"[Init] Instantiating C++ Kernel for model '{self.model_name}'...")
        self.cpp_backend = elastic_body_module.ElasticBody(
            E_base, 0.1, 1e8, 1, nu
        )
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[Init] Output directory ready: {self.output_dir}")

    def _load_and_prepare_mesh(self, filename):
        print(f"[Mesh] Loading from {filename}...")
        try:
            mesh = meshio.read(filename)
            nodes = mesh.points
            cells = mesh.cells_dict['tetra']
        except Exception as e:
            print(f"Error loading mesh with meshio: {e}")
            raise e

        # === [新增] 网格方向修正 (Fix Winding Order) ===
        # 计算每个单元的标架体积 (行列式)
        # Dm = [p1-p0, p2-p0, p3-p0]
        # cells shape: (N, 4) -> nodes indices
        p0 = nodes[cells[:, 0]]
        p1 = nodes[cells[:, 1]]
        p2 = nodes[cells[:, 2]]
        p3 = nodes[cells[:, 3]]
        
        d1 = p1 - p0
        d2 = p2 - p0
        d3 = p3 - p0
        
        # 计算标量三重积 (d1 cross d2) dot d3
        cross = np.cross(d1, d2)
        dets = np.einsum('ij,ij->i', cross, d3)
        
        negative_indices = np.where(dets < 0)[0]
        num_neg = len(negative_indices)
        
        if num_neg > 0:
            print(f"[Mesh] Found {num_neg} inverted elements (Negative Volume). Fixing...")
            # 交换节点 1 和 2 的顺序即可翻转行列式符号
            # cells[idx, 1] <-> cells[idx, 2]
            cells[negative_indices, [1, 2]] = cells[negative_indices, [2, 1]]
            
            # 验证修复
            # 重新计算部分行列式以确认 (可选)
            print(f"[Mesh] All elements re-oriented to positive volume.")
        else:
            print("[Mesh] Mesh orientation is correct (All volumes positive).")

        # === 保存修正后的网格供 C++ 读取 ===
        np.savetxt("nodes.txt", nodes, fmt="%.6f")
        np.savetxt("cells.txt", cells, fmt="%d")
        
        print(f"[Mesh] Generated 'nodes.txt' ({len(nodes)} nodes) and 'cells.txt' ({len(cells)} cells) in CWD.")
        return nodes, cells

    def reset_material(self):
        self.E_field.fill(self.E_base)
        print(f"[Material] Reset to homogeneous E={self.E_base}")

    def add_spherical_inclusion(self, center, radius, E_inclusion):
        center = np.array(center)
        distances = np.linalg.norm(self.centroids - center, axis=1)
        mask = distances <= radius
        count = np.sum(mask)
        if count > 0:
            self.E_field[mask] = E_inclusion
            print(f"[Material] Added sphere at {center}, R={radius}, E={E_inclusion}. Modified {count} cells.")
        else:
            print(f"[Warning] No cells found inside sphere at {center}, R={radius}.")

    def add_cuboid_inclusion(self, p_min, p_max, E_inclusion):
        mask = np.all((self.centroids >= p_min) & (self.centroids <= p_max), axis=1)
        count = np.sum(mask)
        if count > 0:
            self.E_field[mask] = E_inclusion
            print(f"[Material] Added cuboid. Modified {count} cells.")

    def commit_to_cpp(self):
        self.cpp_backend.set_element_modulus(self.E_field)
        print("[System] Material parameters synced to C++ kernel.")

    def export_mesh_data(self):
        np.savetxt(os.path.join(self.output_dir, "nodes.txt"), self.nodes, fmt="%.6f")
        np.savetxt(os.path.join(self.output_dir, "cells.txt"), self.cells, fmt="%d")

    def export_ground_truth_we(self):
        path = os.path.join(self.output_dir, "we.txt")
        np.savetxt(path, self.E_field)
        print(f"[IO] Ground truth stiffness saved to {path}")