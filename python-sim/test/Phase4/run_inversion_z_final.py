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

def run_z_inversion_deep():
    print("=== Phase 4: 深层硬核强力挖掘 (Z-Axis Deep Search) ===")
    
    # 1. 配置
    obs_step_idx = 1
    mesh_path = os.path.join(python_sim_path, "data", "box-3.msh")
    ref_path = os.path.join(python_sim_path, "data", "we-ref.txt")
    
    # 2. 初始化
    # 策略：初始猜测设高一点，避免算法为了省事直接把背景降到底
    E_guess = 100000.0 
    init = ElasticInitializer(mesh_path, "data", E_guess, 0.49)
    init.output_dir = os.path.join(python_sim_path, "data")
    
    # BC: 固定底部
    nodes = init.nodes
    z_min = np.min(nodes[:, 2])
    fixed_indices = np.where(nodes[:, 2] < z_min + 1e-4)[0]
    
    # 3. 运行反演
    try:
        solver = InverseSolver(init, obs_step_idx=obs_step_idx)
        
        # === 核心修改区 ===
        # 1. alpha = 2.5: 这是一个非常激进的值，专门用于压制表面伪影，提炼深层信号。
        # 2. lambda_reg = 5e-15: 进一步减弱正则化，允许深层出现尖锐的突变（球体边界）。
        print("--- Running Deep Search Optimization (Alpha=2.5) ---")
        E_recon = solver.solve_alternating(
            lambda_reg=5e-15,  
            max_iter=50,         # 给它更多时间去慢慢调整深层
            ignore_nodes=fixed_indices,
            alpha=2.5            # <--- 关键！强力放大深层梯度
        )
    except Exception as e:
        print(f"[Error] {e}")
        return

    # 4. 验证与绘图
    if os.path.exists(ref_path):
        E_ref = np.loadtxt(ref_path)
        
        # 计算相关性
        r, _ = stats.pearsonr(E_ref, E_recon)
        print(f"\n=== Validation ===")
        print(f"  Pearson r : {r:.4f}")
        
        # 5. 深度切片分析 (Z-axis Profile)
        # 我们来看看 Z 轴方向的分布，确认是不是把硬度压下去了
        print("[Analysis] Generating Z-axis Profile...")
        cell_centers = np.mean(nodes[init.cells], axis=1)
        z_centers = cell_centers[:, 2] * 1000 # mm
        
        # 按 Z 轴分层取平均值/最大值
        bins = np.linspace(np.min(z_centers), np.max(z_centers), 30)
        bin_z = []
        bin_val_ref = []
        bin_val_recon = []
        
        for i in range(len(bins)-1):
            mask = (z_centers >= bins[i]) & (z_centers < bins[i+1])
            if np.sum(mask) > 0:
                bin_z.append((bins[i] + bins[i+1])/2)
                # 取最大值更能代表每一层里有没有硬核
                bin_val_ref.append(np.max(E_ref[mask]))
                bin_val_recon.append(np.max(E_recon[mask]))
        
        plt.figure(figsize=(8, 6))
        # 交换 XY 轴，让 Z 轴竖着显示，更直观
        plt.plot(bin_val_ref, bin_z, 'k--', label='Ref Max E')
        plt.plot(bin_val_recon, bin_z, 'r-', linewidth=2, label='Recon Max E')
        plt.ylabel('Depth Z (mm) [Bottom=0, Top=Surface]')
        plt.xlabel('Stiffness E (Pa)')
        plt.title(f'Depth Profile (Alpha=2.5): Did we push the inclusion down?')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(init.output_dir, 'z_depth_profile.png'))
        
        # 导出 VTK
        vtk_path = os.path.join(init.output_dir, f"final_result_z_deep.vtk")
        mesh = meshio.Mesh(
            points=nodes,
            cells=[("tetra", init.cells)],
            cell_data={
                "E_Recon": [E_recon],
                "E_Ref": [E_ref]
            }
        )
        mesh.write(vtk_path)
        print(f"[Output] VTK saved: {vtk_path}")
        print("-> 请检查生成的图片 z_depth_profile.png。")
        print("-> 理想情况：红线的峰值不再卡在最顶端，而是向下移动到了中间位置。")

if __name__ == "__main__":
    run_z_inversion_deep()