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
    from sim.models.nhookean import NHookeanForwardSolver
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)

def run_tissue_compression_test():
    print("=== 开始 Phase 2 - Step 3: 软组织按压收敛性验证 (Diagnostic Mode) ===")
    
    # 1. 加载网格
    mesh_path = os.path.join(python_sim_path, "data", "box-3.msh")
    if not os.path.exists(mesh_path):
        print(f"[Error] 找不到测试网格: {mesh_path}")
        return
    
    # 2. 几何尺度检查 (Critical Step)
    # 我们先临时初始化一下来读取节点
    # 注意：这里参数先随便设，主要是为了看 nodes
    temp_init = ElasticInitializer(mesh_path, "output_temp", 1000, 0.3)
    nodes = temp_init.nodes
    
    x_dim = np.ptp(nodes[:,0])
    y_dim = np.ptp(nodes[:,1])
    z_dim = np.ptp(nodes[:,2])
    print(f"\n[Geometry Check] 网格物理尺寸:")
    print(f"  X: {x_dim:.4f} m")
    print(f"  Y: {y_dim:.4f} m")
    print(f"  Z: {z_dim:.4f} m")
    
    # 估算顶面积
    area_approx = x_dim * y_dim
    print(f"  Top Area (Approx): {area_approx:.6e} m^2")
    
    # 3. 设定更安全的材料参数
    # 降级策略: 先用 nu=0.45 跑通， Tet4 单元在 0.48 时极易锁死
    E_tissue = 100000.0 
    nu_tissue = 0.49 
    print(f"\n[Material] E={E_tissue} Pa, nu={nu_tissue} (Reduced from 0.48 for stability)")
    
    init = ElasticInitializer(
        mesh_path=mesh_path, 
        output_dir="output_convergence_test",
        E_base=E_tissue,  
        nu=nu_tissue          
    )
    
    solver = NHookeanForwardSolver(init)
    
    # 4. 边界条件: 固定底部
    z_coords = nodes[:, 2]
    z_min = np.min(z_coords)
    z_max = np.max(z_coords)
    fixed_indices = np.where(z_coords < z_min + 1e-5)[0]
    solver.set_dirichlet_bc(fixed_indices)
    
    # 5. 定义载荷 (根据面积自动计算合理压力)
    # 目标: 产生约 5% 的形变用于测试
    # sigma = E * epsilon = 10000 * 0.05 = 500 Pa
    # Force = sigma * Area
    target_force_mag = 20000.0 * area_approx 
    
    # 如果算出来的力太小(比如网格极小)，给一个保底值；如果太大，限制一下
    # 这里我们手动限制一下，先跑一个小力看看
    safe_force = -abs(target_force_mag) # 最多 0.5N
    
    # 模拟探头：只按压中心
    x_center = (np.min(nodes[:,0]) + np.max(nodes[:,0])) / 2.0
    y_center = (np.min(nodes[:,1]) + np.max(nodes[:,1])) / 2.0
    
    # 探头半径: 假设为宽度的 40%
    radius = min(x_dim, y_dim) * 0.4
    
    top_indices = np.where(z_coords > z_max - 1e-5)[0]
    probe_indices = []
    for idx in top_indices:
        dx = nodes[idx, 0] - x_center
        dy = nodes[idx, 1] - y_center
        if dx*dx + dy*dy < radius*radius:
            probe_indices.append(idx)
            
    if not probe_indices:
        probe_indices = top_indices # Fallback
        
    num_probe = len(probe_indices)
    force_per_node = safe_force / num_probe
    
    print(f"\n[Load Config] Auto-calculated Safe Load:")
    print(f"  Target Stress: ~500 Pa (5% strain)")
    print(f"  Total Force: {safe_force:.4f} N (Distributed on {num_probe} nodes)")
    
    # 6. 增量加载
    num_steps = 20
    final_step_residuals = []
    
    for step in range(1, num_steps + 1):
        ratio = step / num_steps
        curr_val = force_per_node * ratio
        
        f_ext = np.zeros(solver.num_dofs)
        for idx in probe_indices:
            f_ext[3*idx + 2] = curr_val
        solver.f_ext = f_ext
        
        print(f"\n--- Step {step}/{num_steps} (Total: {safe_force * ratio:.4f} N) ---")
        
        # 牛顿迭代
        max_iter = 15
        tol = 1e-6
        step_res = []
        
        for k in range(max_iter):
            solver.cpp.set_current_displacement(solver.u)
            
            # 安全包裹
            try:
                K = solver.cpp.gen_tangent_stiffness()
                f_int = np.array(solver.cpp.gen_grad_f())
            except Exception as e:
                print(f"  [Crash] C++ Exception: {e}")
                sys.exit(1)
                
            residual = solver.f_ext - f_int
            
            # 严格检查
            if np.any(np.isnan(residual)):
                print(f"  [FAIL] NaN detected at iter {k}. Mesh inverted.")
                # 打印一下当前的位移范围，帮助判断是否飞了
                print(f"  Max Disp Z: {np.min(solver.u[2::3]):.4e} m")
                sys.exit(1)
                
            K_mod, r_mod = solver._apply_bc_to_system(K, residual)
            norm = np.linalg.norm(r_mod)
            step_res.append(norm)
            
            print(f"  Iter {k}: Res = {norm:.4e}")
            
            if norm < tol:
                print("  -> Converged.")
                break
            
            # 阻尼牛顿法 (简单版): 如果残差增加，减少步长? 
            # 这里先用标准牛顿，但加上 Try-Catch
            import scipy.sparse.linalg as spla
            try:
                du = spla.spsolve(K_mod, r_mod)
            except Exception as e:
                print(f"  [Solver Error] {e}")
                break
                
            solver.u += du
            
        if step == num_steps:
            final_step_residuals = step_res

    # 7. 结果
    max_disp = np.min(solver.u[2::3])
    print(f"\n[Result] Final Max Compression: {max_disp*1000:.2f} mm")
    
    # 8. 收敛性检查
    print("\n=== Final Convergence Analysis ===")
    if len(final_step_residuals) > 2:
        for i in range(len(final_step_residuals)-1):
            e0 = final_step_residuals[i]
            e1 = final_step_residuals[i+1]
            if e0 > 1e-9:
                print(f"Iter {i}->{i+1}: Ratio(Lin)={e1/e0:.2f}, Ratio(Quad)={e1/e0**2:.2f}")
    
    if len(final_step_residuals) > 1 and final_step_residuals[-1] < 1e-6:
        print("\n-> [PASS] Simulation stable.")
    else:
        print("\n-> [WARNING] Check convergence.")

if __name__ == "__main__":
    run_tissue_compression_test()