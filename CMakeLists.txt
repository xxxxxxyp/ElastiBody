cmake_minimum_required(VERSION 3.18)

# 1. 强制禁用 vcpkg 对 Python 的干扰
set(VCPKG_FIND_PACKAGE_Python3_SKIP_WRAPPER ON)
set(VCPKG_FIND_PACKAGE_Python_SKIP_WRAPPER ON)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

project(TactileElastography LANGUAGES C CXX CUDA)

# --- 核心修改：手动定义 Python3 和 pybind11 ---

# 2. 定位 Conda 路径
set(CONDA_ENV "C:/Users/xxxxxxyp/anaconda3/envs/elastography")

# 3. 手动创建 Python3::Module 目标
# 这样就不再依赖 find_package(Python3)
if(NOT TARGET Python3::Module)
    add_library(Python3::Module INTERFACE IMPORTED)
    set_target_properties(Python3::Module PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CONDA_ENV}/include"
        # 确认一下文件名，3.8 通常是 python38.lib，也可能是 python3.lib
        INTERFACE_LINK_LIBRARIES "${CONDA_ENV}/libs/python38.lib"
    )
    message(STATUS "Manually created Python3::Module target using Conda paths.")
endif()

# 4. 手动创建 pybind11::module 目标
if(NOT TARGET pybind11::module)
    add_library(pybind11::module INTERFACE IMPORTED)
    set_target_properties(pybind11::module PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CONDA_ENV}/include" # Conda 的 pybind11 头文件通常就在这
    )
    message(STATUS "Manually created pybind11::module target.")
endif()

# --- 处理 vcpkg 依赖 (Ipopt, NLopt 等) ---

find_package(PkgConfig REQUIRED)
pkg_check_modules(IPOPT REQUIRED IMPORTED_TARGET ipopt)
find_package(NLopt CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
# --- 定义编译目标 ---

add_library(elastic_body_module MODULE 
    bindings.cpp 
    ElasticBody.cpp
)

target_compile_definitions(elastic_body_module PRIVATE
    EIGEN_USE_MKL_ALL
    NUM_THREADS=16
)

target_include_directories(elastic_body_module PRIVATE
    include
    "${CONDA_ENV}/include"
    "${CONDA_ENV}/Library/include"  # 关键：解决 pybind11.h 丢失
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/include"  # 关键：解决 mkl_pardiso.h 丢失
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 链接
target_link_libraries(elastic_body_module PRIVATE
    pybind11::module
    Python3::Module
    PkgConfig::IPOPT
    "C:/vcpkg/installed/x64-windows/lib/nlopt.lib"
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    CUDA::cudart
    OpenMP::OpenMP_CXX
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/mkl_rt.lib"
)

# 属性
set_target_properties(elastic_body_module PROPERTIES
    PREFIX ""
    SUFFIX ".pyd"
    CUDA_ARCHITECTURES "native"
)