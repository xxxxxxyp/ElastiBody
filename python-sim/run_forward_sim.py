import env_loader
import numpy as np
import os
from sim.geometry.initializer import ElasticInitializer
from sim.models.nhookean import NHookeanForwardSolver

def main():
    # 1. 初始化场景
    print("=== Phase 1: Scene Initialization ===")
    mesh_path = "data/box-3.msh"
    if not os.path.exists(mesh_path):
        print(f"Error: {mesh_path} not found.")
        return

    # 初始化器 (Phase 2 代码)
    initializer = ElasticInitializer(mesh_path, output_dir="data-sampling", E_base=200000.0)
    
    # 获取网格尺寸信息
    nodes = initializer.nodes
    z_coords = nodes[:, 2]
    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    print(f"Mesh Z-range: [{z_min:.4f}, {z_max:.4f}]")
    
    # 添加硬球
    # center = np.mean(nodes, axis=0)
    # initializer.add_spherical_inclusion(center=center, radius=0.0025, E_inclusion=100000.0)
    initializer.commit_to_cpp()
    
    # 导出 Ground Truth we.txt
    initializer.export_ground_truth_we()

    # 2. 配置求解器
    print("\n=== Phase 2: Solver Configuration ===")
    solver = NHookeanForwardSolver(initializer)
    
    # 设定边界条件: 固定底面 (z <= z_min + tolerance)
    tol = 1e-6
    bottom_indices = np.where(z_coords <= z_min + tol)[0]
    solver.set_dirichlet_bc(bottom_indices)
    
    # 设定载荷: 压顶面
    # 找到顶面节点
    top_indices = np.where(z_coords >= z_max - tol)[0]
    print(f"Load Application: Found {len(top_indices)} nodes on top surface.")
    
    # 3. 执行仿真循环 (增量加载)
    print("\n=== Phase 3: Time Stepping ===")
    
    total_steps = 100
    
    # [修正] 物理计算合理的载荷
    # 目标：产生约 10% 的压缩量 (Strain = 0.1)
    # F_total = E * A * Strain = 5000 * (0.01*0.01) * 0.1 = 0.05 N
    # 分摊到每个节点:
    target_strain = 0.2 # 目标 20% 形变
    mesh_area = (np.max(nodes[:,0]) - np.min(nodes[:,0])) * (np.max(nodes[:,1]) - np.min(nodes[:,1]))
    estimated_total_force = 5000.0 * mesh_area * target_strain 
    
    nodes_count_top = len(top_indices)
    max_force_per_node = - (estimated_total_force / nodes_count_top) # 负号表示向下
    
    print(f"Physics Check:")
    print(f"  Target Total Force: {estimated_total_force:.4f} N")
    print(f"  Force per Node:     {max_force_per_node:.6f} N (Applied to {nodes_count_top} nodes)")
    
    for step in range(total_steps):
        # 计算当前步的载荷系数
        ratio = (step + 1) / total_steps
        current_force_val = max_force_per_node * ratio
        
        # 构建外力向量
        f_ext = np.zeros(initializer.num_nodes * 3)
        for node_idx in top_indices:
            f_ext[3*node_idx + 2] = current_force_val
            
        # 求解
        try:
            solver.solve_static_step(f_ext, step_index=step)
        except RuntimeError as e:
            print(f"[Critical Error] Simulation diverged at step {step}: {e}")
            break
        
    print("\n=== Simulation Complete ===")
    print(f"Data generated in {initializer.output_dir}")
    print("You can now run the inverse solver or visualization tools.")

if __name__ == "__main__":
    main()