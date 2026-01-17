import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

def run_lateral_inversion():
    print("=== Phase 4: 最终反向反演 (Lateral Model) ===")
    
    # 1. 配置
    model_name = "box-3"
    obs_step_idx = 10
    mesh_path = os.path.join(python_sim_path, "data", "box-3.msh")
    ref_path = os.path.join(python_sim_path, "data", "we-ref.txt")
    
    # 2. 初始化
    print(f"[Action] Initializing (Frame {obs_step_idx})...")
    # 初始猜测：根据 we-ref 的均值 (550k)，我们猜一个中间值，比如 200k
    # 避免从太软(5k)或太硬(5M)开始，减少震荡
    E_guess = 200000.0 
    nu = 0.49
    
    init = ElasticInitializer(mesh_path, "data", E_guess, nu)
    init.output_dir = os.path.join(python_sim_path, "data")
    init.model_name = model_name
    
    try:
        solver = InverseSolver(init, obs_step_idx=obs_step_idx)
    except FileNotFoundError:
        print("[Error] Data not found.")
        return

    # 3. [关键修正] 设置侧向压缩的边界条件
    nodes = init.nodes
    x_coords = nodes[:, 0]
    x_max = np.max(x_coords)
    
    # 固定右侧面 (X max)
    fixed_indices = np.where(x_coords > x_max - 1e-4)[0]
    print(f"[BC] Fixing Right Face (X > {x_max-1e-4:.4f}): {len(fixed_indices)} nodes")
    
    # 4. 运行反演
    # 参数建议：
    # Reg 1e-13: 保持平滑
    # Alpha 0.6: 适度加强深层
    # Max Iter 40: 给足时间收敛
    E_recon = solver.solve_alternating(
        lambda_reg=1e-13,
        max_iter=40,
        ignore_nodes=fixed_indices, # 这一点至关重要！告诉求解器不要去优化固定点的力平衡
        alpha=0.6
    )
    
    # 5. 加载参考值并对比
    if os.path.exists(ref_path):
        E_ref = np.loadtxt(ref_path)
        
        # 简单统计
        print("\n=== Final Validation ===")
        print(f"  Ref Mean E   : {np.mean(E_ref):.0f} Pa")
        print(f"  Recon Mean E : {np.mean(E_recon):.0f} Pa")
        
        # 计算相关性
        r, _ = stats.pearsonr(E_ref, E_recon)
        print(f"  Pearson r    : {r:.4f}")
        
        if r > 0.5:
            print("  -> [SUCCESS] 强正相关！反演成功。")
        elif r > 0.2:
            print("  -> [PASS] 弱正相关。可能需要微调正则化或初始猜测。")
        else:
            print("  -> [WARN] 相关性依然较低，请检查是否存在其他物理误差。")
            
        # 导出对比 VTK
        vtk_path = os.path.join(init.output_dir, f"final_result_lateral_frame{obs_step_idx}.vtk")
        mesh = meshio.Mesh(
            points=nodes,
            cells=[("tetra", init.cells)],
            cell_data={
                "E_Recon": [E_recon],
                "E_Ref": [E_ref],
                "E_Diff": [E_recon - E_ref]
            }
        )
        mesh.write(vtk_path)
        print(f"\n[Output] Final result saved to: {vtk_path}")
        print("请在 Paraview 中查看 E_Recon，这次它应该和 E_Ref 长得很像了！")

if __name__ == "__main__":
    run_lateral_inversion()