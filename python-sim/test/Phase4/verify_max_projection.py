import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
python_sim_path = os.path.join(project_root, 'python-sim')
if python_sim_path not in sys.path:
    sys.path.append(python_sim_path)

try:
    import env_loader
    from sim.geometry.initializer import ElasticInitializer
    from sim.inverse.solver import InverseSolver
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)

def get_max_projection(x_coords, E_field, num_bins=50):
    """
    将空间沿 X 轴切成若干薄片，取每一片里的 E 最大值。
    这能消除 Y/Z 定位误差的影响，只看 X 轴趋势。
    """
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    bins = np.linspace(x_min, x_max, num_bins + 1)
    
    bin_centers = []
    max_vals = []
    
    for i in range(num_bins):
        # 找到位于当前切片内的节点
        mask = (x_coords >= bins[i]) & (x_coords < bins[i+1])
        if np.sum(mask) > 0:
            # 对应的 E 值（注意：E 是定义在单元上的，这里简单用节点附近的单元近似，或者假设 E_field 长度匹配）
            # 这里我们需要一点技巧：E 是 Cell 属性，x_coords 是 Node 属性。
            # 我们需要计算 Cell Center 的 X 坐标。
            pass 
        bin_centers.append((bins[i] + bins[i+1])/2)
    
    return bins

def verify_max_projection():
    print("=== Phase 4: 最大投影趋势对比 (Max Projection) ===")
    
    # 1. 配置
    model_name = "box-3"
    obs_step_idx = 10
    mesh_path = os.path.join(python_sim_path, "data", "box-3.msh")
    ref_path = os.path.join(python_sim_path, "data", "we-ref.txt")
    
    # 2. 重新运行反演
    print(f"[Action] Re-running inversion (Frame {obs_step_idx})...")
    init = ElasticInitializer(mesh_path, "data", 200000.0, 0.49)
    init.output_dir = os.path.join(python_sim_path, "data")
    
    # BC: 右侧固定
    nodes = init.nodes
    x_max = np.max(nodes[:, 0])
    fixed_indices = np.where(nodes[:, 0] > x_max - 1e-4)[0]
    
    try:
        solver = InverseSolver(init, obs_step_idx=obs_step_idx)
        # 稍微加强一点 alpha，试图把深层的信号放大，让波峰更明显
        E_recon = solver.solve_alternating(
            lambda_reg=1e-13,
            max_iter=15, 
            ignore_nodes=fixed_indices,
            alpha=0.8  # [尝试] 从 0.6 提至 0.8，增强深层权重
        )
    except Exception as e:
        print(f"[Error] Solver failed: {e}")
        return

    # 3. 加载参考
    if not os.path.exists(ref_path): return
    E_ref = np.loadtxt(ref_path)

    # 4. 计算基于 Cell Center 的最大投影
    print("[Analysis] Computing Max Projection along X axis...")
    cell_centers = np.mean(nodes[init.cells], axis=1)
    x_centers = cell_centers[:, 0] * 1000 # mm
    
    # 分桶计算最大值
    bins = np.linspace(np.min(x_centers), np.max(x_centers), 40)
    bin_x = []
    bin_max_ref = []
    bin_max_recon = []
    
    for i in range(len(bins)-1):
        mask = (x_centers >= bins[i]) & (x_centers < bins[i+1])
        if np.sum(mask) > 0:
            bin_x.append((bins[i] + bins[i+1])/2)
            bin_max_ref.append(np.max(E_ref[mask]))
            bin_max_recon.append(np.max(E_recon[mask]))
            
    # 5. 绘图
    plt.figure(figsize=(10, 6))
    
    # 绘制参考值的轮廓
    plt.plot(bin_x, bin_max_ref, 'k--', linewidth=2, label='Reference (Max)')
    plt.fill_between(bin_x, bin_max_ref, color='gray', alpha=0.1)
    
    # 绘制反演值的轮廓
    plt.plot(bin_x, bin_max_recon, 'r-', linewidth=2, marker='o', markersize=4, label='Recon (Max)')
    plt.fill_between(bin_x, bin_max_recon, color='red', alpha=0.1)
    
    plt.xlabel('X Position (mm) [Load at Left, Fixed at Right]')
    plt.ylabel('Max Stiffness E (Pa)')
    plt.title('Max Projection Profile: Does the Peak Align?')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(init.output_dir, 'max_projection_check.png')
    plt.savefig(save_path)
    print(f"\n[Output] Saved plot to: {save_path}")
    print("-> 此次我们提高了 alpha=0.8。请检查红线的波峰是否比之前更明显，位置是否对齐？")

if __name__ == "__main__":
    verify_max_projection()