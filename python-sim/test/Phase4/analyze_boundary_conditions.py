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

def analyze_bc():
    print("=== 观测数据物理场侦探 (BC Detective) ===")
    
    # 1. 加载数据
    obs_step_idx = 10
    data_dir = os.path.join(python_sim_path, "data")
    
    nodes = np.loadtxt(os.path.join(python_sim_path, "nodes.txt"))
    pnt_path = os.path.join(data_dir, f"pnt{obs_step_idx}.txt")
    force_path = os.path.join(data_dir, f"force{obs_step_idx}.txt")
    
    if not os.path.exists(pnt_path):
        print("[Error] Data not found.")
        return

    # 2. 计算位移场
    nodes_obs = np.loadtxt(pnt_path)
    u = (nodes_obs - nodes.flatten()).reshape((-1, 3))
    u_mag = np.linalg.norm(u, axis=1)
    
    # 3. 分析位移与坐标的关系
    print(f"\n[1] 位移分布趋势分析 (Frame {obs_step_idx})")
    print(f"    Max Displacement: {np.max(u_mag)*1000:.3f} mm")
    
    # 计算相关系数
    corr_x = np.corrcoef(nodes[:, 0], u_mag)[0, 1]
    corr_y = np.corrcoef(nodes[:, 1], u_mag)[0, 1]
    corr_z = np.corrcoef(nodes[:, 2], u_mag)[0, 1]
    
    print(f"    Correlation with X: {corr_x:.4f} (若接近 -1/1，说明沿 X 变化)")
    print(f"    Correlation with Y: {corr_y:.4f}")
    print(f"    Correlation with Z: {corr_z:.4f} (我们之前的仿真假定是高正相关)")
    
    # 4. 推断固定端 (位移趋近于 0 的节点)
    # 阈值设为最大位移的 5%
    fix_threshold = np.max(u_mag) * 0.05
    fixed_indices = np.where(u_mag < fix_threshold)[0]
    fixed_nodes = nodes[fixed_indices]
    
    print(f"\n[2] 固定端推断 (Displacement < {fix_threshold*1000:.3f} mm)")
    print(f"    Found {len(fixed_indices)} potential fixed nodes.")
    
    if len(fixed_nodes) > 0:
        print("    Fixed Nodes Bounding Box:")
        print(f"      X: {np.min(fixed_nodes[:,0]):.4f} ~ {np.max(fixed_nodes[:,0]):.4f}")
        print(f"      Y: {np.min(fixed_nodes[:,1]):.4f} ~ {np.max(fixed_nodes[:,1]):.4f}")
        print(f"      Z: {np.min(fixed_nodes[:,2]):.4f} ~ {np.max(fixed_nodes[:,2]):.4f}")
        
        # 自动判定逻辑
        x_span = np.ptp(nodes[:, 0])
        z_span = np.ptp(nodes[:, 2])
        
        mean_fix_x = np.mean(fixed_nodes[:, 0])
        mean_fix_z = np.mean(fixed_nodes[:, 2])
        
        print("\n[3] 自动诊断建议:")
        if np.abs(mean_fix_z - np.min(nodes[:, 2])) < 0.1 * z_span:
            print("    -> 看起来是 [底部固定] (Bottom Fixed)。与之前假设一致，需检查为何分布不同。")
        elif np.abs(mean_fix_x - np.min(nodes[:, 0])) < 0.1 * x_span:
            print("    -> 看起来是 [左侧固定] (Left/Min-X Fixed)。这就解释了 X 轴梯度！")
        elif np.abs(mean_fix_x - np.max(nodes[:, 0])) < 0.1 * x_span:
            print("    -> 看起来是 [右侧固定] (Right/Max-X Fixed)。")
        else:
            print("    -> 固定模式复杂，请看图。")

    # 5. 分析力的方向
    f_vec = np.loadtxt(force_path).reshape((-1, 3))
    total_force = np.sum(f_vec, axis=0)
    print(f"\n[4] 外力方向分析 (Force Vector)")
    print(f"    Total Force: {total_force}")
    if np.abs(total_force[2]) > np.abs(total_force[0]):
        print("    -> 主要受力方向: Z 轴")
    else:
        print("    -> 主要受力方向: X 轴")

    # 6. 绘图验证
    plt.figure(figsize=(12, 5))
    
    # 子图 1: U_mag vs X
    plt.subplot(1, 2, 1)
    plt.scatter(nodes[:, 0]*1000, u_mag*1000, c=u_mag, cmap='viridis', s=10)
    plt.xlabel('X Coordinate (mm)')
    plt.ylabel('Displacement (mm)')
    plt.title('Disp Magnitude vs X (Observed)')
    plt.grid(True)
    
    # 子图 2: U_mag vs Z
    plt.subplot(1, 2, 2)
    plt.scatter(nodes[:, 2]*1000, u_mag*1000, c=u_mag, cmap='viridis', s=10)
    plt.xlabel('Z Coordinate (mm)')
    plt.ylabel('Displacement (mm)')
    plt.title('Disp Magnitude vs Z (Observed)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'bc_detective_frame10.png'))
    print(f"\n[Output] Analysis plot saved to {os.path.join(data_dir, 'bc_detective_frame10.png')}")

if __name__ == "__main__":
    analyze_bc()