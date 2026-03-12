import sys
import os
import numpy as np

# 路径配置：与现有 Phase4 脚本保持一致，确保可以直接导入 python-sim 下的项目模块。
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
python_sim_path = os.path.join(project_root, "python-sim")
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


def load_physics_masks(state_idx, data_dir):
    """
    从 A 矩阵提取 Dirichlet 边界 (固定节点)
    从 B 矩阵提取 观测掩码 (表面可见节点)
    文件格式: row col value。由于每一行代表一个自由度 (DOF)，节点索引 = DOF // 3
    """
    A_path = os.path.join(data_dir, f"A-{state_idx}.txt")
    B_path = os.path.join(data_dir, f"B-{state_idx}.txt")

    def extract_nodes_from_coo(filepath):
        if not os.path.exists(filepath):
            return np.array([], dtype=int)
        data = np.loadtxt(filepath)
        if data.ndim == 1 and len(data) >= 2:
            data = data.reshape(1, -1)
        if data.size == 0:
            return np.array([], dtype=int)

        dofs = data[:, 0].astype(int)
        return np.unique(dofs // 3)

    fixed_nodes = extract_nodes_from_coo(A_path)
    observed_nodes = extract_nodes_from_coo(B_path)

    return fixed_nodes, observed_nodes


if __name__ == "__main__":
    print("=== 12工况联合反演启动 ===")

    # 1. 初始化路径与模型
    data_dir = os.path.join(python_sim_path, "data")
    mesh_path = os.path.join(data_dir, "box-3.msh")
    model_name = "box-3"

    # 按需求优先尝试 E_guess 关键字；若当前初始化器仍使用 E_base 命名，
    # 则回退到兼容写法，避免顶层脚本因接口命名差异而无法运行。
    try:
        init = ElasticInitializer(mesh_path, "data", E_guess=50000.0, nu=0.49)
    except TypeError:
        init = ElasticInitializer(mesh_path, "data", E_base=50000.0, nu=0.49)

    # 与 run_real_data_inversion.py 保持一致：显式指向 python-sim/data 下的实测数据目录。
    init.output_dir = data_dir
    init.model_name = model_name

    # 2. 定义工况列表并逐工况提取真实物理边界
    obs_steps = list(range(1, 13))
    ignore_nodes_list = []
    for state_idx in obs_steps:
        fixed_nodes, observed_nodes = load_physics_masks(state_idx, data_dir)
        ignore_nodes_list.append(fixed_nodes)
        print(
            f"[BC] State {state_idx}: detected {len(fixed_nodes)} fixed nodes, "
            f"{len(observed_nodes)} observed nodes."
        )

    # 3. 实例化联合求解器
    solver = InverseSolver(init, obs_step_list=obs_steps)

    # 4. 启动多工况联合反演
    E_recon = solver.solve_alternating(
        lambda_reg=1e-15,
        max_iter=20,
        ignore_nodes_list=ignore_nodes_list,
        alpha=0.6,
    )

    # 5. 安全加载参考值：如果不存在或尺寸不匹配，则退化为全零参考场。
    ref_path = os.path.join(data_dir, "we-ref.txt")
    E_ref = np.zeros_like(E_recon)
    if os.path.exists(ref_path):
        E_ref_loaded = np.loadtxt(ref_path)
        if np.size(E_ref_loaded) == np.size(E_recon):
            E_ref = np.asarray(E_ref_loaded, dtype=np.float64).flatten()
        else:
            print(
                f"[Warn] Reference field size mismatch: got {np.size(E_ref_loaded)}, "
                f"expected {np.size(E_recon)}. Falling back to zeros."
            )
    else:
        print("[Warn] we-ref.txt not found, filling E_Ref with zeros.")

    # 6. 导出结果：同时写出 TXT 和 VTK，便于数值检查与可视化。
    E_recon_safe = np.asarray(E_recon, dtype=np.float64).flatten()
    E_ref_safe = np.asarray(E_ref, dtype=np.float64).flatten()

    txt_path = os.path.join(data_dir, "joint_inversion_12states.txt")
    np.savetxt(txt_path, E_recon_safe)
    print(f"[Output] Text result saved to {txt_path}")

    vtk_path = os.path.join(data_dir, "joint_inversion_12states.vtk")
    mesh = meshio.Mesh(
        points=init.nodes,
        cells=[("tetra", init.cells)],
        cell_data={
            "E_Recon": [E_recon_safe],
            "E_Ref": [E_ref_safe],
            "E_Diff": [E_recon_safe - E_ref_safe],
        },
    )
    mesh.write(vtk_path)
    print(f"[Output] VTK result saved to {vtk_path}")
