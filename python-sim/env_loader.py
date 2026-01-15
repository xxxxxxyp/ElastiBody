# env_loader.py
import os
import sys

# === 这里写死你的绝对路径 ===
DLL_DIRS = [
    r"C:\vcpkg\installed\x64-windows\bin", 
    r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
]
# CMake 编译输出目录
BUILD_DIR = r"C:\Projects\Elastography with K-tensor\TactileElastography-plus\Elasticbody\build"

# === 自动注入逻辑 ===
def setup():
    # 1. 加载 DLL
    for p in DLL_DIRS:
        if os.path.exists(p):
            os.add_dll_directory(p)
    
    # 2. 添加搜索路径
    if BUILD_DIR not in sys.path:
        sys.path.append(BUILD_DIR)

# 自动执行
setup()