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
    from sim.geometry.initializer import ElasticInitializer
    from sim.inverse.solver import InverseSolver
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)

def calculate_centroid(E_field, nodes, cells, threshold):
    """计算高硬度区域的加权质心"""
    # 找出高于阈值的单元
    indices = np.where(E_field > threshold)[0]
    if len(indices) == 0:
        return None
    
    # 获取这些单元的几何中心
    cell_centers = np.mean(nodes[cells[indices]], axis=1)
    weights = E_field[indices]
    
    # 加权平均
    centroid = np.average(cell_centers, axis=0, weights=weights)
    return centroid

def verify_distribution():
    print("=== Phase 4: 分布一致性深度验证 (Distribution Verification) ===")
    
    # 1. 配置与重跑反演 (Re-run Inversion)
    model_name = "box-3"
    mesh_path = os.path.join(python_sim_path, "data", "box-3.msh")
    ref_path = os.path.join(python_sim_path, "data", "we-ref.txt")
    obs_step_idx = 10
    
    if not os.path.exists(mesh_path) or not os.path.exists(ref_path):
        print("[Error] Mesh or Ref file missing.")
        return

    print(f"[Action] Re-running inversion (Frame {obs_step_idx}, Reg=1e-20)...")
    init = ElasticInitializer(mesh_path, "data", 50000.0, 0.49)
    init.output_dir = os.path.join(python_sim_path, "data")
    init.model_name = model_name
    
    try:
        solver = InverseSolver(init, obs_step_idx=obs_step_idx)
    except:
        print("[Error] Data missing.")
        return

    nodes = init.nodes
    fixed_indices = np.where(nodes[:, 2] < np.min(nodes[:, 2]) + 1e-4)[0]
    
    # 使用您的参数
    E_recon = solver.solve_alternating(
        lambda_reg=1e-15,
        max_iter=20,
        ignore_nodes=fixed_indices,
        alpha=0.7
    )
    E_ref = np.loadtxt(ref_path)

    # 2. 计算统计指标
    print("\n=== Quantitative Metrics ===")
    
    # A. 全局相关性 (Pearson Correlation)
    pearson_r, _ = stats.pearsonr(E_ref, E_recon)
    print(f"1. Pearson Correlation (r): {pearson_r:.4f}")
    if pearson_r > 0.8: print("   -> [PASS] 强线性相关，分布趋势一致。")
    else: print("   -> [WARN] 相关性较弱。")

    # B. 空间重叠度 (Dice Coefficient)
    # 自动阈值: 均值 + 1.0 * 标准差
    thresh_ref = np.mean(E_ref) + 1.0 * np.std(E_ref)
    thresh_recon = np.mean(E_recon) + 1.0 * np.std(E_recon)
    
    mask_ref = E_ref > thresh_ref
    mask_recon = E_recon > thresh_recon
    
    intersection = np.sum(mask_ref & mask_recon)
    size_ref = np.sum(mask_ref)
    size_recon = np.sum(mask_recon)
    dice = 2.0 * intersection / (size_ref + size_recon)
    
    print(f"2. Dice Similarity (DSC)  : {dice:.4f}")
    print(f"   (Threshold: Ref > {thresh_ref:.0f}, Recon > {thresh_recon:.0f})")

    # C. 几何定位精度 (Centroid Error)
    cent_ref = calculate_centroid(E_ref, nodes, init.cells, thresh_ref)
    cent_recon = calculate_centroid(E_recon, nodes, init.cells, thresh_recon)
    
    if cent_ref is not None and cent_recon is not None:
        dist_error = np.linalg.norm(cent_ref - cent_recon) * 1000 # 转为 mm
        print(f"3. Centroid Error         : {dist_error:.2f} mm")
        print(f"   Ref Center   : {cent_ref}")
        print(f"   Recon Center : {cent_recon}")
    else:
        print("3. Centroid Error: [Failed] Could not identify ROI.")

    # 3. 可视化绘图
    print("\n[Action] Generating Plots...")
    save_dir = init.output_dir
    
    # Plot 1: 散点图 (Correlation Scatter)
    plt.figure(figsize=(8, 6))
    plt.scatter(E_ref, E_recon, alpha=0.5, s=10, c='blue', label='Cells')
    
    # 拟合线
    m, b = np.polyfit(E_ref, E_recon, 1)
    plt.plot(E_ref, m*E_ref + b, color='red', linestyle='--', label=f'Fit: y={m:.2f}x+{b:.0f}')
    
    plt.title(f'Correlation Analysis (r={pearson_r:.2f})')
    plt.xlabel('Reference E (Pa)')
    plt.ylabel('Reconstructed E (Pa)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'correlation_scatter.png'))
    print(f"  -> Saved: {os.path.join(save_dir, 'correlation_scatter.png')}")

    # Plot 2: 线轮廓对比 (Line Profile)
    # 选取穿过 Reference 质心的 X 方向直线
    if cent_ref is not None:
        # 找到所有单元中心
        cell_centers = np.mean(nodes[init.cells], axis=1)
        
        # 定义“直线”：Y 和 Z 接近质心的单元
        tol = 0.002 # 2mm 宽度的条带
        mask_line = (np.abs(cell_centers[:, 1] - cent_ref[1]) < tol) & \
                    (np.abs(cell_centers[:, 2] - cent_ref[2]) < tol)
        
        if np.sum(mask_line) > 5:
            indices_line = np.where(mask_line)[0]
            # 按 X 坐标排序
            sorted_order = np.argsort(cell_centers[indices_line, 0])
            sorted_indices = indices_line[sorted_order]
            
            x_vals = cell_centers[sorted_indices, 0] * 1000 # mm
            line_ref = E_ref[sorted_indices]
            line_recon = E_recon[sorted_indices]
            
            # 归一化 (0-1) 以对比形状
            norm_ref = (line_ref - np.min(line_ref)) / (np.ptp(line_ref) + 1e-6)
            norm_recon = (line_recon - np.min(line_recon)) / (np.ptp(line_recon) + 1e-6)
            
            plt.figure(figsize=(10, 5))
            plt.plot(x_vals, norm_ref, 'k--', linewidth=2, label='Reference (Norm)')
            plt.plot(x_vals, norm_recon, 'r-', linewidth=2, label='Recon (Norm)')
            plt.fill_between(x_vals, norm_recon, alpha=0.2, color='red')
            
            plt.title('Normalized Line Profile Comparison (Shape Check)')
            plt.xlabel('X Position (mm)')
            plt.ylabel('Normalized Stiffness (0-1)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'line_profile.png'))
            print(f"  -> Saved: {os.path.join(save_dir, 'line_profile.png')}")
        else:
            print("  -> [Skip] Not enough cells on the center line for profile.")

    plt.close('all')
    print("=== Verification Complete ===")

if __name__ == "__main__":
    verify_distribution()