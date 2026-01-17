import sys
import os
import numpy as np

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
python_sim_path = os.path.join(project_root, 'python-sim')
if python_sim_path not in sys.path:
    sys.path.append(python_sim_path)

try:
    import env_loader
    import meshio
    from sim.geometry.initializer import ElasticInitializer
    from sim.inverse.solver import InverseSolver
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)

def run_analysis_with_reference():
    print("=== Phase 4: 实测数据反演 (Debug Mode) ===")
    
    # 1. 配置
    model_name = "box-3"
    mesh_path = os.path.join(python_sim_path, "data", "box-3.msh")
    ref_path = os.path.join(python_sim_path, "data", "we-ref.txt")
    obs_step_idx = 1  # 您的帧索引
    
    # 2. 初始化
    print(f"[Action] Loading Mesh & Data (Frame {obs_step_idx})...")
    E_guess = 50000.0
    nu = 0.49
    
    init = ElasticInitializer(mesh_path, "data", E_guess, nu)
    init.output_dir = os.path.join(python_sim_path, "data")
    init.model_name = model_name
    
    try:
        solver = InverseSolver(init, obs_step_idx=obs_step_idx)
    except FileNotFoundError:
        print("[Error] pnt/force data not found.")
        return

    # BC
    nodes = init.nodes
    z_min = np.min(nodes[:, 2])
    fixed_indices = np.where(nodes[:, 2] < z_min + 1e-4)[0]
    
    # 3. 运行反演
    print("[Action] Starting Solver...")
    # 注意：这里我把 lambda_reg 改回了 1e-14，1e-20 可能太不稳定
    # 如果您坚持用 1e-20 也可以，但容易出现数值触底
    E_recon = solver.solve_alternating(
        lambda_reg=1e-15, 
        max_iter=20,
        ignore_nodes=fixed_indices,
        alpha=0.6
    )
    
    # 4. 加载参考值 (用于排查是否是 Ref 全为 200)
    E_ref = np.zeros_like(E_recon)
    if os.path.exists(ref_path):
        E_ref = np.loadtxt(ref_path)
        print(f"\n[Debug] E_Ref Stats: Min={np.min(E_ref):.1f}, Max={np.max(E_ref):.1f}")
        if np.allclose(E_ref, 200.0) or np.mean(E_ref) < 3000:
             print("  -> [Warning] 参考值 E_Ref 看起来很小或全是 200！请检查 we-ref.txt")
    else:
        print("[Warn] we-ref.txt not found, filling Ref with zeros.")

    # 5. 写入前的终极检查 (Sanity Check)
    print("\n=== Final Pre-Save Check ===")
    print(f"E_Recon (Memory): Min={np.min(E_recon):.2f}, Max={np.max(E_recon):.2f}, Mean={np.mean(E_recon):.2f}")
    
    if np.max(E_recon) < 3000:
        print("  -> [ALARM] 内存中的 E_Recon 本身就很小！问题出在反演过程，而不是写入过程。")
    else:
        print("  -> [OK] 内存数据正常。")

    # 6. 安全导出
    # 强制转为 float64 numpy 数组
    E_recon_safe = np.array(E_recon, dtype=np.float64).flatten()
    E_ref_safe = np.array(E_ref, dtype=np.float64).flatten()
    
    # 6a. 备份为 TXT
    txt_path = os.path.join(init.output_dir, f"debug_E_recon_frame{obs_step_idx}.txt")
    np.savetxt(txt_path, E_recon_safe)
    print(f"[Output] Text backup saved to {txt_path}")

    # 6b. 导出 VTK
    vtk_filename = f"comparison_frame{obs_step_idx}_debug.vtk"
    vtk_path = os.path.join(init.output_dir, vtk_filename)
    
    mesh = meshio.Mesh(
        points=nodes,
        cells=[("tetra", init.cells)],
        cell_data={
            "E_Recon": [E_recon_safe],
            "E_Ref": [E_ref_safe],
            "E_Diff": [E_recon_safe - E_ref_safe]
        }
    )
    mesh.write(vtk_path)
    print(f"[Output] VTK saved to {vtk_path}")
    print("  -> 请在 Paraview 中确保 'Color By' 选择的是 'E_Recon' 而不是 'E_Ref'！")

if __name__ == "__main__":
    run_analysis_with_reference()