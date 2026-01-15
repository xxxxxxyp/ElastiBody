#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>  // 如果你的代码使用 Eigen，必须包含此头文件
#include <pybind11/stl.h>    // 如果你的代码使用 STL 容器（如 std::vector），必须包含此头文件
#include "ElasticBody.h"       // 替换成包含 ElasticBody 类定义的文件名


namespace py = pybind11;

PYBIND11_MODULE(elastic_body_module, m) {
    py::class_<ElasticBody>(m, "ElasticBody")
        .def(py::init<double, double, double, int, double>(),
             py::arg("E"), py::arg("lowerbound"), py::arg("upperbound"),py::arg("size"),py::arg("nv"))

        .def("load_data", &ElasticBody::load_data)
        .def("gen_grad_f", &ElasticBody::gen_grad_f)
        .def("gen_K", &ElasticBody::gen_K)
        .def("gen_due", &ElasticBody::gen_due)
        .def("eqn", &ElasticBody::eqn)
        .def("jac", &ElasticBody::jac)

        .def("set_current_displacement", &ElasticBody::set_current_displacement, 
             "Update the current node positions based on displacement vector u.")
             
        .def("set_element_modulus", &ElasticBody::set_element_modulus, 
             "Set the Young's modulus for each element.")
             
        .def("gen_tangent_stiffness", &ElasticBody::gen_tangent_stiffness, 
             "Compute the tangent stiffness matrix based on the CURRENT configuration (nodes_rt).")

        .def("gen_unit_geometric_forces", &ElasticBody::gen_unit_geometric_forces,
             "Compute the 12xN matrix of unit geometric forces (K-tensor basis) for inverse analysis.")
        
        .def_readwrite("f_sensor", &ElasticBody::f_sensor)
        .def_readwrite("we", &ElasticBody::we)
        .def_readwrite("due", &ElasticBody::due)
        .def_readwrite("due_sum", &ElasticBody::due_sum)
        .def_readwrite("B_list", &ElasticBody::B_list)
        .def_readwrite("A_list", &ElasticBody::A_list)
        .def_readwrite("nodes_rt_list", &ElasticBody::nodes_rt_list)
        
        .def_readonly("num_nodes", &ElasticBody::num_nodes)
        .def_readonly("num_cells", &ElasticBody::num_cells);
}
