import sys
import os
import numpy as np
import meshio  # [Fix] 需要导入 meshio 生成 .msh 文件

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
python_sim_path = os.path.join(project_root, 'python-sim')
if python_sim_path not in sys.path:
    sys.path.append(python_sim_path)

try:
    import env_loader
    from sim.geometry.initializer import ElasticInitializer
    from sim.models.nhookean import NHookeanForwardSolver
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)

def generate_patch_mesh_2x2x2(filename_nodes="nodes.txt", filename_cells="cells.txt", filename_msh="patch_test.msh"):
    """
    生成一个 2x2x2 的由四面体组成的立方体网格。
    同时生成 .txt (给 C++) 和 .msh (给 Python)
    """
    print(f"[Mesh] Generating {filename_msh} and text files...")
    
    # 1. 生成节点 grid 3x3x3
    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 1, 3)
    z = np.linspace(0, 1, 3)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    nodes = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    
    # 2. 生成单元
    cells_list = []
    
    def get_idx(i, j, k):
        return i*9 + j*3 + k

    # 遍历 2x2x2 个小立方体，每个切分为 6 个四面体
    for i in range(2):
        for j in range(2):
            for k in range(2):
                p0 = get_idx(i,   j,   k)
                p1 = get_idx(i+1, j,   k)
                p2 = get_idx(i+1, j+1, k)
                p3 = get_idx(i,   j+1, k)
                p4 = get_idx(i,   j,   k+1)
                p5 = get_idx(i+1, j,   k+1)
                p6 = get_idx(i+1, j+1, k+1)
                p7 = get_idx(i,   j+1, k+1)
                
                # Cube -> 6 Tets 切分方案
                cells_list.append([p0, p1, p2, p6])
                cells_list.append([p0, p2, p3, p6])
                cells_list.append([p0, p3, p7, p6])
                cells_list.append([p0, p7, p4, p6])
                cells_list.append([p0, p4, p5, p6])
                cells_list.append([p0, p5, p1, p6])

    cells = np.array(cells_list, dtype=int)

    # 3. 写入 C++ 需要的文本文件
    np.savetxt(filename_nodes, nodes, fmt="%.6f")
    np.savetxt(filename_cells, cells, fmt="%d")
    
    # 4. [Fix] 写入 Python 需要的 .msh 文件
    # meshio 需要 cells 格式为 [("type", data)]
    mesh = meshio.Mesh(
        points=nodes,
        cells=[("tetra", cells)]
    )
    mesh.write(filename_msh)
    
    return nodes, filename_msh

def run_patch_test():
    print("=== 开始 Phase 2 - Step 2: 片测试 (Patch Test) ===")
    
    # 1. 准备网格 (同时生成 txt 和 msh)
    nodes, msh_path = generate_patch_mesh_2x2x2()
    print(f"[Info] 网格已生成: 27 nodes, 48 cells -> {msh_path}")
    
    # 2. 初始化求解器
    # 这里传入真实的 msh_path
    init = ElasticInitializer(
        mesh_path=msh_path, 
        output_dir="output_patch_test",
        E_base=1000.0,
        nu=0.3
    )
    
    solver = NHookeanForwardSolver(init)
    
    # 3. 定义线性位移场 (Linear Patch Function)
    # u_x = 0.1 * x + 0.05 * y
    # u_y = 0.1 * y + 0.05 * z
    # u_z = 0.1 * z + 0.05 * x
    def exact_displacement(n):
        x, y, z = n[0], n[1], n[2]
        ux = 0.1 * x + 0.05 * y
        uy = 0.1 * y + 0.05 * z
        uz = 0.1 * z + 0.05 * x
        return np.array([ux, uy, uz])

    # 4. 施加 Dirichlet 边界条件
    boundary_indices = []
    internal_indices = []
    
    u_exact = np.zeros(3 * 27)
    
    eps = 1e-5
    for i in range(27):
        n = nodes[i]
        u_vec = exact_displacement(n)
        u_exact[3*i : 3*i+3] = u_vec
        
        # 判断是否在边界
        if (n[0] < eps or n[0] > 1-eps or 
            n[1] < eps or n[1] > 1-eps or 
            n[2] < eps or n[2] > 1-eps):
            boundary_indices.append(i)
        else:
            internal_indices.append(i)
            
    print(f"[Info] 边界节点数: {len(boundary_indices)} (应为 26)")
    print(f"[Info] 内部节点数: {len(internal_indices)} (应为 1)")
    
    # 设置求解器的边界条件
    solver.set_dirichlet_bc(boundary_indices)
    
    # [Critical] 将边界节点的位移值注入到 solver.u 中作为强制约束
    for idx in boundary_indices:
        solver.u[3*idx : 3*idx+3] = u_exact[3*idx : 3*idx+3]
        
    # 5. 运行求解
    print("[Action] 开始 Newton-Raphson 求解...")
    # force_input=0.0 因为这是位移驱动的
    solver.solve_static_step(force_input=0.0, tol=1e-6, max_iter=20)
    
    # 6. 验证内部节点结果
    print("\n=== 结果验证 ===")
    passed = True
    for idx in internal_indices:
        u_sol = solver.u[3*idx : 3*idx+3]
        u_ref = u_exact[3*idx : 3*idx+3]
        diff = np.linalg.norm(u_sol - u_ref)
        
        print(f"Node {idx} (Center):")
        print(f"  Exact: {u_ref}")
        print(f"  Solved: {u_sol}")
        print(f"  Error: {diff:.2e}")
        
        if diff > 1e-5: # 稍微放宽一点点宽容度给数值误差
            passed = False
            
    if passed:
        print("\n-> [PASS] Patch Test 通过！求解器组装正确。")
    else:
        print("\n-> [FAIL] Patch Test 失败。内部节点位移不匹配。")

if __name__ == "__main__":
    run_patch_test()