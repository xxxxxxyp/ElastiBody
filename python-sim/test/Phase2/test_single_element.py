import sys
import os
import numpy as np
import scipy.sparse as sp

# ==========================================
# 1. 路径配置
# ==========================================
# 假设脚本位于 test/Phase2/ 下，我们需要向上两级找到 python-sim
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
python_sim_path = os.path.join(project_root, 'python-sim')

if python_sim_path not in sys.path:
    sys.path.append(python_sim_path)

try:
    import env_loader
    from sim.geometry.initializer import ElasticInitializer
    # 尝试加载 C++ 模块 (通过 env_loader)
    # 注意：ElasticBody 类通常直接由 module 导出，或者在 sim 内部被封装
    # 这里我们直接访问 ElasticBody 类，如果不成功则需检查 env_loader
    import elastic_body_module
except ImportError as e:
    print(f"[Error] 无法加载仿真环境或C++模块: {e}")
    print(f"请检查路径: {python_sim_path}")
    sys.exit(1)

# ==========================================
# 2. 辅助函数：解析解计算
# ==========================================
def analytic_neohookean_forces(F, E, nu, vol, Dm_inv):
    """
    Python 端独立计算 NeoHookean 节点力
    注意：这里的公式必须与 C++ 中的实现逻辑完全一致
    """
    # Lame 参数转换
    mu = E / (2 * (1 + nu))
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    
    J = np.linalg.det(F)
    FinvT = np.linalg.inv(F).T
    logJ = np.log(J)
    
    # 第一类 Piola-Kirchhoff 应力张量 P
    # P = mu * (F - F^{-T}) + lambda * logJ * F^{-T}
    P = mu * (F - FinvT) + lam * logJ * FinvT
    
    # 转换为节点力
    # C++代码逻辑: H = -volume * P * Dm_inv.T
    # forces_1,2,3 是 H 的列
    # force_0 = -(f1+f2+f3)
    H = -vol * P @ Dm_inv.T
    
    f0 = -H[:, 0] - H[:, 1] - H[:, 2]
    f1 = H[:, 0]
    f2 = H[:, 1]
    f3 = H[:, 2]
    
    return np.concatenate([f0, f1, f2, f3])

# ==========================================
# 3. 核心测试流程
# ==========================================
def run_single_element_test():
    print("=== 开始 Phase 2: 单单元物理一致性测试 ===")
    
    # --- A. 生成临时网格 (单位四面体) ---
    # 节点: 原点 + 三个轴上的单位点
    nodes_data = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    # 单元: 0-1-2-3
    cells_data = np.array([[0, 1, 2, 3]], dtype=int)
    
    np.savetxt("nodes.txt", nodes_data, fmt="%.6f")
    np.savetxt("cells.txt", cells_data, fmt="%d")
    print("[Info] 临时网格文件已生成 (nodes.txt, cells.txt)")

    # --- B. 初始化 C++ 对象 ---
    E = 1000.0
    nu = 0.3
    # ElasticBody(E, lowerbound, upperbound, size, nv)
    # size 是 total_times，这里设为 1
    sim = elastic_body_module.ElasticBody(E, 0.0, 0.0, 1, nu)
    
    # 这一步非常重要：C++ 内部硬编码读取当前目录下的 mesh 文件
    # 且需要在构造后显式调用 load_data 或其他初始化逻辑? 
    # 看 C++ 代码构造函数里已经调用了 read_mesh。
    
    # 验证体积
    # 单位四面体体积 = 1/6 = 0.166666...
    # 我们没法直接从 python 获取 volume 数组，但可以通过力来侧面验证
    vol_ref = 1.0 / 6.0
    
    Dm = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]).T # Dm cols are edges. Edge 0: 1-0=(1,0,0), etc.
    Dm_inv = np.linalg.inv(Dm)

    # --- C. 施加形变测试 (单轴拉伸) ---
    # 将节点 1 (x=1) 拉伸到 x=1.2 (lambda=1.2)
    u = np.zeros(12)
    stretch_x = 0.2
    u[3] = stretch_x # Node 1 x-dof (index 3)
    
    # 1. 注入位移
    sim.set_current_displacement(u)
    
    # 2. C++ 计算内力
    f_cpp = np.array(sim.gen_grad_f())
    
    # 3. Python 解析解计算
    # 构造变形梯度 F
    # x = X + u. Node 1 moved from (1,0,0) to (1.2,0,0)
    # F = dxdX = diag(1.2, 1, 1)
    F = np.diag([1.2, 1.0, 1.0])
    
    f_analytic = analytic_neohookean_forces(F, E, nu, vol_ref, Dm_inv)
    
    # 4. 对比结果
    print("\n[Check 1] 内力计算验证 (Internal Force Validation):")
    diff = f_cpp - f_analytic
    error_norm = np.linalg.norm(diff)
    
    print(f"  F_cpp (Node 1 x): {f_cpp[3]:.6f}")
    print(f"  F_ana (Node 1 x): {f_analytic[3]:.6f}")
    print(f"  Max Diff: {np.max(np.abs(diff)):.2e}")
    print(f"  L2 Error: {error_norm:.2e}")
    
    if error_norm < 1e-8:
        print("  -> [PASS] 内力物理公式验证通过")
    else:
        print("  -> [FAIL] 内力与解析解不符，请检查 C++ 公式实现")
        
    # --- D. 切线刚度矩阵验证 (Finite Difference Check) ---
    print("\n[Check 2] 切线刚度矩阵验证 (Tangent Stiffness Check):")
    
    # 1. 获取解析/C++ 刚度矩阵
    K_cpp = sim.gen_tangent_stiffness()
    # K_cpp 是稀疏矩阵，转为 dense
    if sp.issparse(K_cpp):
        K_cpp = K_cpp.toarray()
        
    # 2. 有限差分计算 K_fd
    # K_ij = d(f_i)/d(u_j)
    epsilon = 1e-7
    K_fd = np.zeros((12, 12))
    
    # 计算参考力
    sim.set_current_displacement(u)
    f0 = np.array(sim.gen_grad_f())
    
    for j in range(12):
        u_perturb = u.copy()
        u_perturb[j] += epsilon
        
        sim.set_current_displacement(u_perturb)
        f_perturb = np.array(sim.gen_grad_f())
        
        col = (f_perturb - f0) / epsilon
        K_fd[:, j] = col
        
    # 3. 对比
    K_diff = K_cpp - K_fd
    # 忽略非常小的值的相对误差，主要看绝对误差
    k_error_norm = np.linalg.norm(K_diff)
    rel_error = k_error_norm / (np.linalg.norm(K_cpp) + 1e-10)
    
    print(f"  K_cpp norm: {np.linalg.norm(K_cpp):.4f}")
    print(f"  K_fd  norm: {np.linalg.norm(K_fd):.4f}")
    print(f"  K Matrix Abs Error: {k_error_norm:.2e}")
    print(f"  K Matrix Rel Error: {rel_error:.2e}")
    
    # 刚度矩阵通常误差会比力大一点，但应控制在 1e-4 ~ 1e-5 级别 (取决于 epsilon)
    if rel_error < 1e-4:
        print("  -> [PASS] 切线刚度矩阵验证通过 (Consistent)")
    else:
        print("  -> [FAIL] 切线刚度矩阵可能不正确 (Inconsistent)")
        # 打印差异最大的元素位置
        max_idx = np.unravel_index(np.argmax(np.abs(K_diff)), K_diff.shape)
        print(f"    Max Diff at ({max_idx[0]}, {max_idx[1]}): {K_diff[max_idx]:.6f}")

    # 清理临时文件
    # if os.path.exists("nodes.txt"): os.remove("nodes.txt")
    # if os.path.exists("cells.txt"): os.remove("cells.txt")

if __name__ == "__main__":
    run_single_element_test()