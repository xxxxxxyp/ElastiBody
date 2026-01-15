# ElastiBody

这是一个以 C++ 实现的 Neo-Hookean 弹性体仿真模块，并通过 `pybind11` 暴露到 Python，用于正向仿真、反演与力学指标分析。

## 项目结构与文件说明

```
.
├── CMakeLists.txt
├── ElasticBody.cpp
├── ElasticBody.h
├── bindings.cpp
├── python-sim/
│   ├── env_loader.py
│   ├── run_forward_sim.py
│   ├── run_inverse_analysis.py
│   ├── compute_metrics.py
│   ├── data/
│   │   ├── box-3.msh
│   │   ├── A-*.txt / B-*.txt
│   │   ├── pnt*.txt / force*.txt
│   │   └── box-3-we-forward.txt
│   ├── sim/
│   │   ├── __init__.py
│   │   ├── geometry/
│   │   │   ├── __init__.py
│   │   │   └── initializer.py
│   │   ├── models/
│   │   │   └── nhookean.py
│   │   └── inverse/
│   │       ├── regularization.py
│   │       ├── sensitivity.py
│   │       └── solver.py
│   ├── nodes.txt
│   ├── cells.txt
│   └── debug_force_check.vtk
├── .vscode/
│   ├── c_cpp_properties.json
│   └── tasks.json
├── .gitignore
└── vc140.pdb
```

### 根目录文件

- **CMakeLists.txt**：CMake 构建脚本，定义 `elastic_body_module` 的编译方式，包含 Windows/Conda/vcpkg/Intel MKL/CUDA 的依赖路径配置。
- **ElasticBody.h**：`ElasticBody` 类的头文件，声明网格读取、力学计算、刚度矩阵与反演接口等核心 API。
- **ElasticBody.cpp**：`ElasticBody` 的实现文件，包含读取网格、计算 Piola 应力、装配刚度矩阵、求解位移与灵敏度矩阵等算法逻辑。
- **bindings.cpp**：`pybind11` 绑定代码，将 `ElasticBody` 暴露为 Python 模块 `elastic_body_module`。
- **.gitignore**：Git 忽略规则。
- **vc140.pdb**：Windows/MSVC 生成的调试符号文件（调试时使用）。

### python-sim/（Python 驱动与数据样例）

- **env_loader.py**：在 Windows 上注入 DLL 搜索路径与 CMake 构建目录，保证 Python 能找到编译后的 `.pyd`。
- **run_forward_sim.py**：正向仿真脚本，基于外部 `sim` 包执行增量加载并导出数据。
- **run_inverse_analysis.py**：反演示例脚本，调用 `sim.inverse` 执行参数反演并输出 VTK 结果。
- **compute_metrics.py**：后处理脚本，计算变形梯度与主伸长率并导出 VTK 可视化。
- **data/**：示例网格与仿真输出数据（A/B 稀疏矩阵、力与位移观测、`box-3.msh` 等）。
- **sim/**：Python 端算法实现包。
  - **geometry/initializer.py**：读取 mesh、修正四面体方向、初始化材料场并创建 C++ 后端对象。
  - **models/nhookean.py**：前向求解器，执行 Newton 迭代并导出 A/B/force/pnt 数据。
  - **inverse/solver.py**：反演主流程（灵敏度加权与正则化迭代）。
  - **inverse/sensitivity.py**：构建基于 C++ 单元力的灵敏度矩阵。
  - **inverse/regularization.py**：构建单元邻接拉普拉斯矩阵用于平滑正则化。
- **nodes.txt / cells.txt**：四面体网格节点与单元索引数据，供 C++/Python 读取。
- **debug_force_check.vtk**：调试用 VTK 结果文件，便于在 ParaView 中查看力场或变形分布。

### .vscode/（本地开发配置）

- **c_cpp_properties.json**：VS Code 的 C/C++ IntelliSense 配置（包含 Windows SDK 与 vcpkg include 路径）。
- **tasks.json**：VS Code 编译任务模板（以 `cl.exe` 构建当前文件）。
