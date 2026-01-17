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

def analyze_bc_frame1():
    print("=== 1号数据物理场侦探 (BC Detective - Frame 1) ===")
    
    # 1. 配置
    obs_step_idx = 1  # 用户指定的垂直按压数据
    data_dir = os.path.join(python_sim_path, "data")
    nodes = np.loadtxt(os.path.join(python_sim_path, "nodes.txt"))
    
    pnt_path = os.path.join(data_dir, f"pnt{obs_step_idx}.txt")
    force_path = os.path.join(data_dir, f"force{obs_step_idx}.txt")
    
    if not os.path.exists(pnt_path):
        print(f"[Error] {pnt_path} not found.")
        return

    # 2. 分析力 (Force)
    f_vec = np.loadtxt(force_path).reshape((-1, 3))
    total_force = np.sum(f_vec, axis=0)
    print(f"\n[1] 外力方向 (Force)")
    print(f"    Total Force: {total_force} N")
    
    if np.abs(total_force[2]) > np.abs(total_force[0]) and total_force[2] < 0:
        print("    -> [符合预期] 主要受力为 Z 轴向下 (垂直按压)。")
    else:
        print(f"    -> [注意] 受力方向似乎不是纯 Z 轴向下？请检查。")

    # 3. 分析位移 (Displacement)
    nodes_obs = np.loadtxt(pnt_path)
    u = (nodes_obs - nodes.flatten()).reshape((-1, 3))
    u_mag = np.linalg.norm(u, axis=1)
    
    print(f"\n[2] 位移分布 (Displacement)")
    print(f"    Max Displacement: {np.max(u_mag)*1000:.3f} mm")
    
    # 相关性分析
    corr_z = np.corrcoef(nodes[:, 2], u_mag)[0, 1]
    print(f"    Corr(u, Z): {corr_z:.4f}")
    
    if corr_z > 0.8:
        print("    -> [符合预期] 位移随 Z 轴高度增加而增大 (顶部动，底部不动)。")
    
    # 4. 寻找固定端
    fix_threshold = np.max(u_mag) * 0.05
    fixed_indices = np.where(u_mag < fix_threshold)[0]
    fixed_nodes = nodes[fixed_indices]
    
    print(f"\n[3] 固定端推断 (Fixed Nodes)")
    print(f"    Found {len(fixed_indices)} fixed nodes.")
    
    if len(fixed_nodes) > 0:
        z_min = np.min(nodes[:, 2])
        mean_fix_z = np.mean(fixed_nodes[:, 2])
        if np.abs(mean_fix_z - z_min) < 0.002:
            print("    -> [确认] 底部固定 (Bottom Fixed)。")
        else:
            print(f"    -> 固定端 Z 坐标: {mean_fix_z:.4f} (Min Z: {z_min:.4f})")
            
    # 5. 绘图验证
    plt.figure(figsize=(6, 5))
    plt.scatter(nodes[:, 2]*1000, u_mag*1000, c=u_mag, cmap='viridis', s=10)
    plt.xlabel('Z Coordinate (mm)')
    plt.ylabel('Displacement (mm)')
    plt.title(f'Frame {obs_step_idx}: Disp vs Z (Vertical Press?)')
    plt.grid(True)
    plt.savefig(os.path.join(data_dir, f'bc_detective_frame{obs_step_idx}.png'))
    print(f"\n[Output] Plot saved to data/bc_detective_frame{obs_step_idx}.png")

if __name__ == "__main__":
    analyze_bc_frame1()