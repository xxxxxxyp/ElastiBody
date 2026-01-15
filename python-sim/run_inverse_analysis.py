import env_loader
import numpy as np
from sim.geometry.initializer import ElasticInitializer
from sim.inverse.solver import InverseSolver

def main():
    # 1. 初始化
    mesh_path = "data/box-3.msh"
    initializer = ElasticInitializer(mesh_path, output_dir="data-sampling", E_base=20000.0)
    
    # 2. 识别需要剔除的固定节点 (底面)
    # 这必须与 run_forward_sim.py 中的固定逻辑一致
    nodes = initializer.nodes
    z_coords = nodes[:, 2]
    z_min = np.min(z_coords)
    # 找到底面节点索引
    bottom_indices = np.where(z_coords <= z_min + 1e-5)[0]
    print(f"[Config] Identifying {len(bottom_indices)} bottom nodes to exclude from equilibrium eq.")

    # 3. 实例化反向求解器
    solver = InverseSolver(initializer, obs_step_idx=9)
    
    # 4. 运行反演
    # 传入 ignore_nodes=bottom_indices
    # 尝试稍微减小正则化 lambda_reg=1e-9 以允许更锐利的边界
    E_recon = solver.solve_alternating(lambda_reg=1e-20, max_iter=100, ignore_nodes=bottom_indices)
    
    # 5. 导出
    solver.export_result()
    
    # 6. 统计与可视化
    print("\n=== Inversion Result Statistics ===")
    print(f"Max E: {np.max(E_recon):.2f} (Target: ~50000)")
    print(f"Min E: {np.min(E_recon):.2f} (Target: ~5000)")
    print(f"Mean E: {np.mean(E_recon):.2f}")

    # 导出 VTK
    import meshio
    vtk_path = f"data-sampling/box-3/result_inverse.vtk"
    mesh = meshio.Mesh(
        points=initializer.nodes,
        cells=[("tetra", initializer.cells)],
        cell_data={"E_Reconstructed": [E_recon]}
    )
    mesh.write(vtk_path)
    print(f"Visualization saved to {vtk_path}")

if __name__ == "__main__":
    main()