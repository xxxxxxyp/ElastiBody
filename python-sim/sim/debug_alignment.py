import numpy as np
import meshio
import os

# 假设您已经有了初始化好的 solver 对象 (adj_solver)
# 直接调用这个函数

def debug_alignment(adj_solver):
    print("=== 启动坐标系与残差诊断 (Alignment Debug) ===")
    
    # 1. 运行一次正向仿真 (基于均匀初始硬度)
    # 设一个均匀的硬度，比如 400kPa
    s_vec = np.ones(adj_solver.num_cells) * 0.8 # 假设 E_scaling=500k -> 400k
    E_physical = s_vec * adj_solver.E_scaling
    
    print("  Running Forward Simulation...")
    u_sim = adj_solver._solve_forward(E_physical)
    
    # 2. 获取观测数据
    u_obs = adj_solver.u_obs
    
    # 3. 计算残差 (Residual) = Sim - Obs
    # 如果坐标系对齐，残差应该集中在受力点附近。
    # 如果不对齐，残差会像“太极图”一样，一边正一边负。
    diff = u_sim - u_obs
    
    # 计算标量误差 (用于可视化幅度)
    diff_mag = np.linalg.norm(diff.reshape(-1, 3), axis=1)
    
    # 4. 计算一次原始梯度 (Raw Gradient)
    # 看看第一步梯度到底指哪里
    loss, grad = adj_solver._objective_and_gradient(s_vec)
    
    print("  Exporting Debug VTK...")
    
    # 导出到一个 VTK 文件里，方便对比
    # 我们把所有向量都存进去
    mesh = meshio.Mesh(
        points=adj_solver.init.nodes,
        cells=[("tetra", adj_solver.init.cells)],
        point_data={
            "U_Sim": u_sim.reshape(-1, 3),
            "U_Obs": u_obs.reshape(-1, 3),
            "Diff_Vector": diff.reshape(-1, 3),
            "Diff_Mag": diff_mag
        },
        cell_data={
            "Gradient_Start": [grad] # 这是导致“远处单元变硬”的罪魁祸首
        }
    )
    
    output_path = os.path.join(adj_solver.init.output_dir, "debug_alignment.vtk")
    mesh.write(output_path)
    print(f"=== 诊断完成，结果已保存至 {output_path} ===")
    print("请务必在 Paraview 中打开此文件，使用 Glyph 箭头查看 U_Sim 和 U_Obs 的方向是否一致！")

# 使用方法：
# solver = AdjointSolver(...)
# debug_alignment(solver)