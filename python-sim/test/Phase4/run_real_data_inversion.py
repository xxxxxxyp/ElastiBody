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
    import meshio # 确保引入 meshio
    from sim.geometry.initializer import ElasticInitializer
    from sim.inverse.solver import InverseSolver
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)

def run_analysis_with_reference():
    print("=== Phase 4: 实测数据反演与基准对比 (Inversion vs Reference) ===")
    
    # 1. 配置
    model_name = "box-3"
    mesh_path = os.path.join(python_sim_path, "data", "box-3.msh")
    ref_path = os.path.join(python_sim_path, "data", "we-ref.txt")
    
    # [User Param] 使用您指定的参数
    obs_step_idx = 1 
    
    # 2. 检查文件
    if not os.path.exists(mesh_path):
        print(f"[Error] Mesh not found: {mesh_path}")
        return
    if not os.path.exists(ref_path):
        print(f"[Error] Reference file not found: {ref_path}")
        print("请确保 we-ref.txt 位于 python-sim/data/ 目录下")
        return

    # 3. 初始化环境 (用于读取网格信息)
    print(f"[Action] Initializing environment...")
    E_guess = 50000.0
    nu = 0.49
    
    init = ElasticInitializer(mesh_path, "data", E_guess, nu)
    init.output_dir = os.path.join(python_sim_path, "data")
    init.model_name = model_name

    # 4. [新增功能] 优先加载并导出 Reference VTK
    # 这样即使反演失败，您也能先看到参考值的样子
    print(f"\n[Data] Loading reference distribution from {ref_path}...")
    try:
        E_ref = np.loadtxt(ref_path)
    except Exception as e:
        print(f"[Error] Failed to load reference file: {e}")
        return

    # 检查维度匹配
    if E_ref.shape[0] != init.num_cells:
        print(f"[Error] Ref size {E_ref.shape[0]} != Num Cells {init.num_cells}")
        return

    # === 核心修改：立即保存独立的 Reference VTK ===
    ref_vtk_path = os.path.join(init.output_dir, "reference_ground_truth.vtk")
    mesh_ref = meshio.Mesh(
        points=init.nodes,
        cells=[("tetra", init.cells)],
        cell_data={"E_Ref": [E_ref]}
    )
    mesh_ref.write(ref_vtk_path)
    print(f"[Output] Independent Reference VTK saved to: {ref_vtk_path}")
    print("---------------------------------------------------------------")

    # 5. 执行反演
    print(f"[Action] Running inversion for Frame {obs_step_idx}...")
    try:
        solver = InverseSolver(init, obs_step_idx=obs_step_idx)
    except FileNotFoundError:
        print("[Error] pnt/force data not found.")
        return

    # 底部固定 BC
    nodes = init.nodes
    z_min = np.min(nodes[:, 2])
    fixed_indices = np.where(nodes[:, 2] < z_min + 1e-4)[0]
    
    # 运行反演 (使用您的参数)
    E_recon = solver.solve_alternating(
        lambda_reg=5e-15,
        max_iter=20,
        ignore_nodes=fixed_indices,
        alpha=0.6
    )
    
    # 6. 计算误差指标
    print("\n=== Error Analysis ===")
    
    # 绝对差值
    diff = E_recon - E_ref
    abs_diff = np.abs(diff)
    
    # 统计量
    mae = np.mean(abs_diff) # 平均绝对误差
    rmse = np.sqrt(np.mean(diff**2)) # 均方根误差
    max_err = np.max(abs_diff) # 最大误差
    
    # 相对误差 (L2 Norm)
    norm_ref = np.linalg.norm(E_ref)
    norm_diff = np.linalg.norm(diff)
    rel_error_l2 = norm_diff / (norm_ref + 1e-8)
    
    print(f"  Mean Absolute Error (MAE): {mae:.2f} Pa")
    print(f"  Root Mean Sq Error (RMSE): {rmse:.2f} Pa")
    print(f"  Max Pointwise Error      : {max_err:.2f} Pa")
    print(f"  Relative Error (L2 Norm) : {rel_error_l2*100:.2f}%")
    
    # 区域性对比
    roi_indices = np.where(E_ref > np.mean(E_ref) + np.std(E_ref))[0]
    
    if len(roi_indices) > 0:
        mean_roi_recon = np.mean(E_recon[roi_indices])
        mean_roi_ref = np.mean(E_ref[roi_indices])
        print(f"\n[ROI Analysis] (Hard Inclusion Area)")
        print(f"  Ref Mean E   : {mean_roi_ref:.2f}")
        print(f"  Recon Mean E : {mean_roi_recon:.2f}")
        print(f"  Recovery Rate: {mean_roi_recon/mean_roi_ref*100:.1f}%")

    # 7. 导出对比 VTK (包含 Ref, Recon, Diff)
    vtk_filename = f"comparison_frame{obs_step_idx}.vtk"
    vtk_path = os.path.join(init.output_dir, vtk_filename)
    
    mesh = meshio.Mesh(
        points=nodes,
        cells=[("tetra", init.cells)],
        cell_data={
            "E_Recon": [E_recon],
            "E_Ref": [E_ref],
            "E_Diff": [diff],           # 带符号的差值 (Recon - Ref)
            "E_AbsDiff": [abs_diff]     # 绝对误差
        }
    )
    mesh.write(vtk_path)
    print(f"\n[Output] Comparison VTK saved to {vtk_path}")
    print("  -> E_Diff > 0 表示反演值偏硬，E_Diff < 0 表示偏软")

if __name__ == "__main__":
    run_analysis_with_reference()