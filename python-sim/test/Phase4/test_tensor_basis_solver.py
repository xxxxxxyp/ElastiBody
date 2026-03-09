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
    from sim.geometry.initializer import ElasticInitializer
    # 引入我们刚才写的新求解器
    from sim.inverse.tensor_solver import TensorIsotropicSolver
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)

def test_tensor_solver():
    print("=== 测试: 基于张量基底的新求解器 (TensorIsotropicSolver) ===")
    
    # 1. 准备数据
    mesh_path = os.path.join(python_sim_path, "data", "box-3.msh")
    obs_step_idx = 10  # 假设使用第10帧数据
    data_dir = os.path.join(python_sim_path, "data")
    
    pnt_path = os.path.join(data_dir, f"pnt{obs_step_idx}.txt")
    force_path = os.path.join(data_dir, f"force{obs_step_idx}.txt")
    
    if not os.path.exists(pnt_path):
        print(f"[Skip] 数据文件 {pnt_path} 不存在，请先生成数据。")
        return

    # 2. 初始化
    init = ElasticInitializer(mesh_path, "data", 200000.0, 0.49)
    
    # 加载观测
    nodes0 = init.nodes
    nodes_obs = np.loadtxt(pnt_path)
    u_obs = nodes_obs - nodes0.flatten()
    f_ext = np.loadtxt(force_path)
    
    # 固定底部 BC
    z_min = np.min(nodes0[:, 2])
    fixed_indices = np.where(nodes0[:, 2] < z_min + 1e-4)[0]
    fixed_dofs = []
    for idx in fixed_indices:
        fixed_dofs.extend([3*idx, 3*idx+1, 3*idx+2])
    
    # 3. 实例化新求解器
    print("[Init] Instantiating TensorIsotropicSolver...")
    solver = TensorIsotropicSolver(init, u_obs, f_ext, fixed_dofs)
    
    # 4. 运行一次梯度计算 (Check functionality)
    s_vec = np.ones(init.num_cells)
    print("  Calculating gradient via tensor basis decomposition...")
    try:
        loss, grad = solver._objective_and_gradient(s_vec)
        print(f"  [Success] Loss = {loss:.4e}")
        print(f"  [Success] Gradient Norm = {np.linalg.norm(grad):.4e}")
        print(f"  [Info] Gradient Min/Max = {np.min(grad):.2e} / {np.max(grad):.2e}")
        
        if np.isnan(loss) or np.any(np.isnan(grad)):
            print("  [Fail] NaN detected!")
        else:
            print("  -> C++ 接口调用与投影逻辑工作正常。")
            
    except Exception as e:
        print(f"  [Error] {e}")
        import traceback
        traceback.print_exc()

    # 5. 试运行反演 (Optional)
    print("\n[Run] Running short optimization (5 steps)...")
    try:
        E_recon = solver.solve(s_vec * 200000.0, total_steps=5)
        print("  -> Optimization loop works.")
    except Exception as e:
        print(f"  [Error during solve] {e}")

if __name__ == "__main__":
    test_tensor_solver()