import numpy as np
import os
import meshio

def compute_deformation_metrics(nodes_ref, nodes_def, cells):
    """
    计算所有单元的变形梯度 F 和主伸长率 Lambda
    :param nodes_ref: 参考构型节点坐标 (N, 3)
    :param nodes_def: 变形后节点坐标 (N, 3)
    :param cells: 单元索引 (M, 4)
    :return: 
        principal_stretches: (M, 3) 每个单元的三个主伸长率
        F_tensors: (M, 3, 3) 每个单元的变形梯度张量
    """
    num_cells = len(cells)
    
    # 1. 获取四面体四个顶点的坐标
    # Shape: (M, 4, 3)
    p_ref = nodes_ref[cells] 
    p_def = nodes_def[cells]
    
    # 2. 计算形状矩阵 (Shape Matrices) Dm 和 Ds
    # 选取第0个节点作为原点，计算三条边向量
    # D = [p1-p0, p2-p0, p3-p0] (按列排列)
    # 为了利用 numpy 的 broadcasting，我们构造 (M, 3, 3)
    # transpose(0, 2, 1) 是为了把向量转为列向量
    
    # 参考构型 Dm
    Dm = np.empty((num_cells, 3, 3))
    Dm[:, :, 0] = p_ref[:, 1, :] - p_ref[:, 0, :]
    Dm[:, :, 1] = p_ref[:, 2, :] - p_ref[:, 0, :]
    Dm[:, :, 2] = p_ref[:, 3, :] - p_ref[:, 0, :]
    
    # 变形构型 Ds
    Ds = np.empty((num_cells, 3, 3))
    Ds[:, :, 0] = p_def[:, 1, :] - p_def[:, 0, :]
    Ds[:, :, 1] = p_def[:, 2, :] - p_def[:, 0, :]
    Ds[:, :, 2] = p_def[:, 3, :] - p_def[:, 0, :]
    
    # 3. 计算变形梯度 F = Ds * Dm^-1
    # numpy.linalg.solve(A, B) 求解 AX = B => X = A^-1 B
    # 这里我们要算 F = Ds @ inv(Dm)。
    # 也就是 F @ Dm = Ds。转置后: Dm^T @ F^T = Ds^T
    # 这种线性代数变换有点绕，直接用 matmul @ inv 更直观，虽然 solve 更快
    
    try:
        Dm_inv = np.linalg.inv(Dm)
    except np.linalg.LinAlgError:
        print("Error: Singular elements detected in reference mesh (Zero volume).")
        return None, None
        
    # F: (M, 3, 3)
    F = np.matmul(Ds, Dm_inv)
    
    # 4. 计算右 Cauchy-Green 张量 C = F^T F
    # transpose(0, 2, 1) 对每个矩阵进行转置
    FT = F.transpose(0, 2, 1)
    C = np.matmul(FT, F)
    
    # 5. 特征值分解求主伸长率
    # eigenvalues 返回的 w 可能不按顺序
    eig_vals = np.linalg.eigvalsh(C) # eigvalsh 用于对称矩阵，更快更稳
    
    # 避免数值误差导致的微小负数
    eig_vals = np.clip(eig_vals, 0.0, None)
    
    lambda_stretches = np.sqrt(eig_vals)
    
    return lambda_stretches, F

def main():
    model_name = "box-3"
    data_dir = f"data-sampling/{model_name}"
    
    # 1. 读取参考网格
    try:
        nodes0 = np.loadtxt("nodes.txt")
        cells = np.loadtxt("cells.txt", dtype=int)
    except OSError:
        print("Error: nodes.txt or cells.txt not found in current directory.")
        return

    # 2. 读取最后一步的变形结果
    step_idx = 9 
    pnt_file = os.path.join(data_dir, f"pnt{step_idx}.txt")
    if not os.path.exists(pnt_file):
        print(f"Error: Result file {pnt_file} not found.")
        return
        
    nodes_final_flat = np.loadtxt(pnt_file)
    nodes_final = nodes_final_flat.reshape((-1, 3))
    
    print(f"Computing metrics for step {step_idx}...")
    
    # 3. 计算指标
    stretches, F_tensors = compute_deformation_metrics(nodes0, nodes_final, cells)
    
    if stretches is None:
        return

    # 4. 统计与导出
    # 计算每个单元的最大主伸长率
    max_stretch = np.max(stretches, axis=1)
    min_stretch = np.min(stretches, axis=1)
    
    print("\n=== Deformation Statistics ===")
    print(f"Max Principal Stretch: {np.max(max_stretch):.4f} (Expansion)")
    print(f"Min Principal Stretch: {np.min(min_stretch):.4f} (Compression)")
    print(f"Mean Volume Change (det F): {np.mean(np.linalg.det(F_tensors)):.4f}")
    
    # 导出为 VTK 以便在 Paraview 中查看 (需要 meshio)
    output_vtk = os.path.join(data_dir, f"result_step_{step_idx}.vtk")
    
    # [FIX] 修改字段名：去除空格和括号
    mesh = meshio.Mesh(
        points=nodes_final,
        cells=[("tetra", cells)],
        cell_data={
            "lambda_1": [stretches[:, 0]],
            "lambda_2": [stretches[:, 1]],
            "lambda_3": [stretches[:, 2]],
            "max_stretch": [max_stretch],
            "J_Volume_Ratio": [np.linalg.det(F_tensors)]  # 修改这里: "J (Volume Ratio)" -> "J_Volume_Ratio"
        }
    )
    mesh.write(output_vtk)
    print(f"\n[IO] Visualization file saved to: {output_vtk}")
    print("Tip: Open this .vtk file in Paraview to visualize the internal strain field.")

if __name__ == "__main__":
    main()