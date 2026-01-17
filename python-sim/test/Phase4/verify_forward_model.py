import sys
import os
import numpy as np
import meshio

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
python_sim_path = os.path.join(project_root, 'python-sim')
if python_sim_path not in sys.path:
    sys.path.append(python_sim_path)

try:
    import env_loader
    from sim.geometry.initializer import ElasticInitializer
    from sim.models.nhookean import NHookeanForwardSolver
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)

def verify_forward_consistency():
    print("=== 正向一致性校验 (Forward Consistency Check) ===")
    
    # 1. 配置
    model_name = "box-3"
    obs_step_idx = 10  # 使用第 10 帧 (高信噪比)
    
    data_dir = os.path.join(python_sim_path, "data")
    mesh_path = os.path.join(data_dir, "box-3.msh")
    ref_E_path = os.path.join(data_dir, "we-ref.txt")
    force_path = os.path.join(data_dir, f"force{obs_step_idx}.txt")
    pnt_path = os.path.join(data_dir, f"pnt{obs_step_idx}.txt")
    
    # 2. 检查文件
    for p in [mesh_path, ref_E_path, force_path, pnt_path]:
        if not os.path.exists(p):
            print(f"[Error] File not found: {p}")
            return

    # 3. 加载数据
    print(f"[Data] Loading frame {obs_step_idx} inputs...")
    E_ref = np.loadtxt(ref_E_path)
    f_ext = np.loadtxt(force_path)     # 力的输入
    nodes_obs = np.loadtxt(pnt_path)   # 点的位移输入 (观测到的最终坐标)
    
    # 4. 初始化仿真环境
    # 使用 E_ref 作为材料参数进行初始化
    # 注意：nu 仍然需要假设，通常用 0.49
    nu = 0.49
    print(f"[Init] Setting up simulation with E_ref (Mean: {np.mean(E_ref):.0f} Pa)...")
    
    init = ElasticInitializer(mesh_path, "data", 5000.0, nu) # E_base 随便填，会被覆盖
    init.output_dir = data_dir
    
    # 强制注入参考 E
    init.E_field = E_ref
    init.commit_to_cpp()
    init.cpp_backend.set_element_modulus(E_ref)
    
    # 5. 设置前向求解器
    solver = NHookeanForwardSolver(init)
    
    # 6. 设置边界条件 (BC)
    # 我们需要固定底部，这必须与产生实测数据时的实验条件一致
    nodes = init.nodes
    z_min = np.min(nodes[:, 2])
    fixed_indices = np.where(nodes[:, 2] < z_min + 1e-4)[0]
    solver.set_dirichlet_bc(fixed_indices)
    
    # 7. 运行正向模拟 (Forward Simulation)
    print(f"[Sim] Running forward simulation with Force input...")
    # 我们用 f_ext 驱动，看看算出来的 u 是否等于观测的 u
    solver.solve_static_step(f_ext, step_index=999, tol=1e-6)
    
    # 8. 对比位移 (Displacement Comparison)
    # 计算观测位移 u_obs
    # nodes_obs 是变形后的坐标，nodes 是初始坐标
    # u_obs = current - initial
    u_obs_flattened = nodes_obs - nodes.flatten()
    u_sim_flattened = solver.u
    
    # 转为 (N, 3) 方便计算距离
    u_obs = u_obs_flattened.reshape((-1, 3))
    u_sim = u_sim_flattened.reshape((-1, 3))
    
    # 计算误差
    diff_vec = u_sim - u_obs
    diff_norm = np.linalg.norm(diff_vec, axis=1) # 每个点的误差距离
    
    u_obs_norm = np.linalg.norm(u_obs, axis=1)
    
    # 9. 统计指标
    mae = np.mean(diff_norm)
    max_err = np.max(diff_norm)
    
    # 相对误差 (对于位移接近0的点，相对误差没意义，所以只统计位移较大的点)
    mask_move = u_obs_norm > 1e-5
    if np.sum(mask_move) > 0:
        rel_err = np.mean(diff_norm[mask_move] / u_obs_norm[mask_move])
    else:
        rel_err = 0.0
        
    print("\n=== Consistency Report ===")
    print(f"  Max Simulated Disp : {np.max(np.linalg.norm(u_sim, axis=1))*1000:.3f} mm")
    print(f"  Max Observed Disp  : {np.max(u_obs_norm)*1000:.3f} mm")
    print("-" * 30)
    print(f"  Mean Error (MAE)   : {mae*1000:.3f} mm")
    print(f"  Max Error          : {max_err*1000:.3f} mm")
    print(f"  Relative Error     : {rel_err*100:.2f}% (on moving nodes)")
    
    # 10. 导出 VTK
    vtk_path = os.path.join(data_dir, f"forward_verification_frame{obs_step_idx}.vtk")
    mesh = meshio.Mesh(
        points=nodes,
        cells=[("tetra", init.cells)],
        point_data={
            "u_Simulated": u_sim,
            "u_Observed": u_obs,
            "u_Difference": diff_vec,
            "Error_Magnitude": diff_norm
        },
        cell_data={
            "E_Ref": [E_ref]
        }
    )
    mesh.write(vtk_path)
    print(f"\n[Output] Saved verification VTK to: {vtk_path}")
    print("  -> Paraview Tips:")
    print("     1. 使用 'Warp By Vector' 滤镜，选择 u_Simulated 和 u_Observed 分别查看变形形态。")
    print("     2. 如果形态一致但幅值不同，说明 E_Ref 的数值量级可能与 Force 不匹配。")

if __name__ == "__main__":
    verify_forward_consistency()