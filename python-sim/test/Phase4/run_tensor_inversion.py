import sys
import os
import numpy as np
import meshio

# 1. 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
python_sim_path = os.path.join(project_root, 'python-sim')
if python_sim_path not in sys.path:
    sys.path.append(python_sim_path)

try:
    import env_loader
    from sim.geometry.initializer import ElasticInitializer
    # [关键修改] 引入基于张量基底的新求解器
    from sim.inverse.tensor_solver import TensorIsotropicSolver
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)

def main():
    # === 配置区 ===
    obs_step_idx = 1  # 使用 Frame 1 (垂直按压)
    mesh_filename = "box-3.msh"
    data_dir = os.path.join(python_sim_path, "data")
    
    print(f"--- Loading Data from {data_dir} ---")
    
    # 1. 加载观测数据
    pnt_path = os.path.join(data_dir, f"pnt{obs_step_idx}.txt")
    force_path = os.path.join(data_dir, f"force{obs_step_idx}.txt")
    
    if not os.path.exists(pnt_path):
        print(f"Error: {pnt_path} not found.")
        return

    # 2. 初始化物理环境
    mesh_path = os.path.join(data_dir, mesh_filename)
    # 初始猜测 E_base，TensorSolver 会在此基础上进行背景校准
    E_guess_val = 200000.0
    init = ElasticInitializer(mesh_path, "data", E_guess_val, 0.49)
    
    # 3. 准备数据向量
    nodes0 = init.nodes
    nodes_obs = np.loadtxt(pnt_path)
    u_obs_full = nodes_obs - nodes0.flatten() # 目标位移
    f_ext = np.loadtxt(force_path)
    
    print("!!! Flipping Force Direction (Standard Fix) !!!")
    f_ext = -f_ext
    
    # 4. 确定固定边界 (Dirichlet BC)
    # 逻辑：固定底部 Z_min
    z_min = np.min(nodes0[:, 2])
    fixed_node_indices = np.where(nodes0[:, 2] < z_min + 1e-4)[0]
    
    fixed_dofs = []
    for idx in fixed_node_indices:
        fixed_dofs.extend([3*idx, 3*idx+1, 3*idx+2])
    fixed_dofs = np.array(fixed_dofs, dtype=int)
    
    print(f"[BC] Fixed {len(fixed_node_indices)} nodes at bottom.")

    # === 核心调用区 ===
    
    # 5. [关键修改] 实例化张量求解器
    print("\n[Solver] Initializing TensorIsotropicSolver (Tensor Basis Mode)...")
    # TensorIsotropicSolver 继承自 AdjointSolver，接口保持兼容
    solver = TensorIsotropicSolver(init, u_obs_full, f_ext, fixed_dofs)
    
    # 6. 运行反演
    # TensorIsotropicSolver 的 solve 方法内置了背景校准和动态截断
    # 我们不需要手动传 x0，它会自动计算最佳 s_base
    print("[Solver] Starting Optimization Loop...")
    E_final = solver.solve(total_steps=200)
    
    # 7. 保存结果
    output_vtk = os.path.join(data_dir, f"tensor_result_frame{obs_step_idx}.vtk")
    
    # 加载参考值对比 (如果有)
    ref_path = os.path.join(data_dir, "we-ref.txt")
    cell_data = {
        "E_Recon": [E_final],
        "log_E": [np.log10(E_final + 1.0)] # 辅助查看数量级
    }
    
    if os.path.exists(ref_path):
        E_ref = np.loadtxt(ref_path)
        cell_data["E_Ref"] = [E_ref]
        cell_data["Error"] = [np.abs(E_final - E_ref)]
        cell_data["Error_Rel"] = [np.abs(E_final - E_ref) / (E_ref + 1.0)]
        print(f"\n[Validation] Ref Mean: {np.mean(E_ref):.0f}, Recon Mean: {np.mean(E_final):.0f}")
        
    mesh = meshio.Mesh(
        points=nodes0,
        cells=[("tetra", init.cells)],
        cell_data=cell_data
    )
    mesh.write(output_vtk)
    print(f"Result saved to: {output_vtk}")
    print("提示: 在 Paraview 中，E_Recon 是通过张量基底投影回来的等效标量模量。")

if __name__ == "__main__":
    main()