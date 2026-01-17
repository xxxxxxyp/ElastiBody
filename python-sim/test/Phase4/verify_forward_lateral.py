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

def verify_lateral_model():
    print("=== 正向一致性校验 (Lateral Compression Model) ===")
    
    # 1. 配置
    model_name = "box-3"
    obs_step_idx = 10
    data_dir = os.path.join(python_sim_path, "data")
    
    # 2. 加载原始参考值 (Reset to original)
    ref_E_path = os.path.join(data_dir, "we-ref.txt")
    if not os.path.exists(ref_E_path):
        print(f"[Error] {ref_E_path} not found")
        return
        
    E_ref = np.loadtxt(ref_E_path)
    print(f"[Data] Loaded E_ref (Mean: {np.mean(E_ref):.0f} Pa)")
    
    # 3. 初始化仿真
    mesh_path = os.path.join(data_dir, "box-3.msh")
    # 注意: 实测数据通常泊松比不会正好是 0.49，有时 0.45-0.48 更符合非受限压缩
    # 但我们先保持 0.49 不变
    init = ElasticInitializer(mesh_path, "data", 5000.0, 0.49)
    init.output_dir = data_dir
    
    # 注入原始参考 E
    init.E_field = E_ref
    init.commit_to_cpp()
    init.cpp_backend.set_element_modulus(E_ref)
    
    # 4. 设置求解器与 [修正后的边界条件]
    solver = NHookeanForwardSolver(init)
    
    nodes = init.nodes
    x_coords = nodes[:, 0]
    x_max = np.max(x_coords)
    x_min = np.min(x_coords)
    
    # [CORRECTION] 固定右侧面 (Max X)
    # 根据 Corr(X, u) = -0.99，说明 X 最大处位移最小 -> 固定端
    fixed_indices = np.where(x_coords > x_max - 1e-4)[0]
    print(f"[BC] Fixing Right Face (X > {x_max-1e-4:.4f}): {len(fixed_indices)} nodes")
    solver.set_dirichlet_bc(fixed_indices)
    
    # 5. 加载力并求解
    force_path = os.path.join(data_dir, f"force{obs_step_idx}.txt")
    f_ext = np.loadtxt(force_path)
    
    # 简单检查力的方向是否匹配
    f_vec = f_ext.reshape((-1, 3))
    total_f = np.sum(f_vec, axis=0)
    print(f"[Load] Total Force applied: {total_f} N")
    
    print(f"[Sim] Running simulation...")
    solver.solve_static_step(f_ext, step_index=999, tol=1e-6)
    
    # 6. 对比结果
    pnt_path = os.path.join(data_dir, f"pnt{obs_step_idx}.txt")
    nodes_obs = np.loadtxt(pnt_path)
    u_obs = (nodes_obs - nodes.flatten()).reshape((-1, 3))
    u_sim = solver.u.reshape((-1, 3))
    
    u_obs_norm = np.linalg.norm(u_obs, axis=1)
    u_sim_norm = np.linalg.norm(u_sim, axis=1)
    
    max_obs = np.max(u_obs_norm)
    max_sim = np.max(u_sim_norm)
    
    # 计算新的校准系数
    new_calibration_factor = max_sim / max_obs
    
    print("\n=== Result (Corrected BCs) ===")
    print(f"  Max Observed Disp  : {max_obs*1000:.3f} mm")
    print(f"  Max Simulated Disp : {max_sim*1000:.3f} mm")
    print(f"  Ratio (Sim/Obs)    : {new_calibration_factor:.4f}")
    
    # 7. 导出校准建议
    if 0.8 < new_calibration_factor < 1.2:
        print("\n-> [PASS] 模型吻合！E_ref 无需大幅校准。")
    else:
        print(f"\n-> [ACTION] 建议校准 E_ref。")
        print(f"   请将 E_ref 乘以 {new_calibration_factor:.4f} 以匹配观测位移。")
        
        # 自动生成校准文件
        E_ref_new = E_ref * new_calibration_factor
        save_path = os.path.join(data_dir, "we-ref-calibrated-lateral.txt")
        np.savetxt(save_path, E_ref_new)
        print(f"   已保存校准后的参考值至: {save_path}")

    # 导出 VTK 供肉眼确认分布形态是否一致
    vtk_path = os.path.join(data_dir, f"verify_lateral_frame{obs_step_idx}.vtk")
    mesh = meshio.Mesh(
        points=nodes,
        cells=[("tetra", init.cells)],
        point_data={
            "u_Simulated": u_sim,
            "u_Observed": u_obs,
            "Error_Mag": np.linalg.norm(u_sim - u_obs, axis=1)
        },
        cell_data={"E_Ref": [E_ref]}
    )
    mesh.write(vtk_path)
    print(f"\n[Viz] VTK saved to {vtk_path}. 请在 Paraview 中对比 u_Simulated 和 u_Observed 的矢量方向。")

if __name__ == "__main__":
    verify_lateral_model()