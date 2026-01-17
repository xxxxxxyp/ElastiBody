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
    import meshio
    from sim.geometry.initializer import ElasticInitializer
    from sim.inverse.solver import InverseSolver
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)

def calculate_weighted_centroid(E_field, nodes, cells):
    """计算刚度场的加权质心 (寻找最硬的地方在哪)"""
    # 为了突出硬核，减去背景均值，只计算高出部分
    baseline = np.min(E_field)
    weights = E_field - baseline
    weights = np.maximum(weights, 0) # 确保非负
    
    # 如果太平坦，就用原始值
    if np.sum(weights) < 1e-9:
        weights = E_field
        
    cell_centers = np.mean(nodes[cells], axis=1)
    centroid = np.average(cell_centers, axis=0, weights=weights)
    return centroid

def verify_lateral_distribution():
    print("=== Phase 4: 侧向模型分布诊断 (Lateral Distribution Check) ===")
    
    # 1. 配置
    model_name = "box-3"
    obs_step_idx = 10
    mesh_path = os.path.join(python_sim_path, "data", "box-3.msh")
    ref_path = os.path.join(python_sim_path, "data", "we-ref.txt")
    
    # 2. 重新运行反演 (快速复现内存数据)
    print(f"[Action] Re-running inversion (Frame {obs_step_idx})...")
    init = ElasticInitializer(mesh_path, "data", 200000.0, 0.49)
    init.output_dir = os.path.join(python_sim_path, "data")
    
    # BC: 右侧固定
    nodes = init.nodes
    x_max = np.max(nodes[:, 0])
    fixed_indices = np.where(nodes[:, 0] > x_max - 1e-4)[0]
    
    try:
        solver = InverseSolver(init, obs_step_idx=obs_step_idx)
        E_recon = solver.solve_alternating(
            lambda_reg=1e-13,
            max_iter=10, # 快速跑几步即可复现结果
            ignore_nodes=fixed_indices,
            alpha=0.6
        )
    except Exception as e:
        print(f"[Error] Solver failed: {e}")
        return

    # 3. 加载参考值
    if not os.path.exists(ref_path):
        print("[Error] we-ref.txt missing")
        return
    E_ref = np.loadtxt(ref_path)

    # 4. 计算重心位置 (Centroid of Stiffness)
    print("\n=== Stiffness Centroid Analysis ===")
    cent_ref = calculate_weighted_centroid(E_ref, nodes, init.cells)
    cent_recon = calculate_weighted_centroid(E_recon, nodes, init.cells)
    
    # 转换为 mm
    c_ref_mm = cent_ref * 1000
    c_recon_mm = cent_recon * 1000
    
    print(f"Ref Center (XYZ mm)   : [{c_ref_mm[0]:.2f}, {c_ref_mm[1]:.2f}, {c_ref_mm[2]:.2f}]")
    print(f"Recon Center (XYZ mm) : [{c_recon_mm[0]:.2f}, {c_recon_mm[1]:.2f}, {c_recon_mm[2]:.2f}]")
    
    dist = np.linalg.norm(cent_ref - cent_recon) * 1000
    print(f"Distance Error        : {dist:.2f} mm")
    
    # 5. X轴分布切片分析 (Lateral Profile)
    # 因为是侧向压缩，我们最关心 X 轴上的硬度分布
    print("\n[Analysis] Checking distribution along compression axis (X)...")
    cell_centers = np.mean(nodes[init.cells], axis=1)
    x_centers = cell_centers[:, 0] * 1000
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_centers, E_ref, alpha=0.3, label='Reference', c='gray', s=10)
    plt.scatter(x_centers, E_recon, alpha=0.3, label='Recon', c='red', s=10)
    
    # 绘制移动平均线 (Trend)
    # 按 X 坐标排序
    sort_idx = np.argsort(x_centers)
    x_sorted = x_centers[sort_idx]
    
    # 简单的滑动平均
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    window = 50
    if len(x_sorted) > window:
        y_ref_smooth = moving_average(E_ref[sort_idx], window)
        y_recon_smooth = moving_average(E_recon[sort_idx], window)
        x_smooth = x_sorted[window//2 : -window//2 + 1]
        
        plt.plot(x_smooth, y_ref_smooth, 'k--', lw=3, label='Ref Trend')
        plt.plot(x_smooth, y_recon_smooth, 'r-', lw=3, label='Recon Trend')
    
    plt.xlabel('X Position (mm)')
    plt.ylabel('Stiffness E (Pa)')
    plt.title(f'Stiffness Distribution along Compression Axis (Dist Error: {dist:.1f}mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(init.output_dir, 'lateral_distribution_check.png')
    plt.savefig(save_path)
    print(f"\n[Output] Saved distribution plot to: {save_path}")
    print("-> 请查看生成的图片：红线（Recon）的波峰是否和黑虚线（Ref）在同一个 X 位置？")

if __name__ == "__main__":
    verify_lateral_distribution()