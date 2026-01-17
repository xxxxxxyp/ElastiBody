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

def verify_forward_z():
    print("=== 正向一致性校验 (Z-Axis Compression) ===")
    
    # 1. 配置
    obs_step_idx = 1
    data_dir = os.path.join(python_sim_path, "data")
    mesh_path = os.path.join(data_dir, "box-3.msh")
    ref_path = os.path.join(data_dir, "we-ref.txt")
    
    # 2. 加载参考 E
    if not os.path.exists(ref_path):
        print("we-ref.txt missing")
        return
    E_ref = np.loadtxt(ref_path)
    
    # 3. 初始化
    init = ElasticInitializer(mesh_path, "data", 5000.0, 0.49)
    init.output_dir = data_dir
    init.E_field = E_ref
    init.commit_to_cpp()
    init.cpp_backend.set_element_modulus(E_ref)
    
    # 4. BC: 固定底部 (Z Min)
    nodes = init.nodes
    z_min = np.min(nodes[:, 2])
    fixed_indices = np.where(nodes[:, 2] < z_min + 1e-4)[0]
    
    solver = NHookeanForwardSolver(init)
    solver.set_dirichlet_bc(fixed_indices)
    
    # 5. 加载 Force 1
    force_path = os.path.join(data_dir, f"force{obs_step_idx}.txt")
    f_ext = np.loadtxt(force_path)
    
    print("[Sim] Running Forward Sim...")
    solver.solve_static_step(f_ext, step_index=999, tol=1e-6)
    
    # 6. 对比
    pnt_path = os.path.join(data_dir, f"pnt{obs_step_idx}.txt")
    nodes_obs = np.loadtxt(pnt_path)
    u_obs = (nodes_obs - nodes.flatten()).reshape((-1, 3))
    u_sim = solver.u.reshape((-1, 3))
    
    max_sim = np.max(np.linalg.norm(u_sim, axis=1))
    max_obs = np.max(np.linalg.norm(u_obs, axis=1))
    
    print(f"\n=== Result (Frame {obs_step_idx}) ===")
    print(f"  Max Obs Disp : {max_obs*1000:.3f} mm")
    print(f"  Max Sim Disp : {max_sim*1000:.3f} mm")
    print(f"  Ratio        : {max_sim/max_obs:.4f}")
    
    if 0.5 < max_sim/max_obs < 1.5:
        print("-> [PASS] 物理量级匹配！可以直接反演。")
    else:
        print("-> [WARN] 量级偏差较大，反演结果可能会有比例误差。")

if __name__ == "__main__":
    verify_forward_z()