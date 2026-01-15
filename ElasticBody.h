#ifndef ELASTICBODY_H
#define ELASTICBODY_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/PardisoSupport>
#include <mkl_pardiso.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>
#include <regex>
#include <cmath>
#include <array>
#include <omp.h>
#include <coin-or/IpIpoptApplication.hpp>
#include <coin-or/IpTNLP.hpp>
#include <nlopt.hpp>

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

class ElasticBody {
public:
    int total_times;
    vector<vector<Vector3d>> nodes_rt_list;
    vector<Vector3d> nodes, nodes0, nodes_rt;
    vector<Vector4i> cells;
    VectorXd f, f_temp;
    vector<VectorXd> f_temp_list;
    MatrixXd Klocal;
    std::array<Vector3d, 4> node_pos, node_pos0;
    double mu_o, mu, la_o, la;
    Matrix3d Dm, Ds, Dminv, F, Finv, FinvT, FtF, piola, H, term;
    vector<Matrix3d> Dm_list, Dm_inv_list;
    Matrix<double, 4, 3> gradC;
    VectorXd node_force, node_force_t;
    size_t num_nodes, num_cells;
    double J, logJ, Ic, energy, nv, stepn;
    std::array<Matrix3d, 12> dD, dF, dP, dH;
    SparseMatrix<double> K, A;
    vector<SparseMatrix<double>> A_list, B_list;
    VectorXd we;
    VectorXd volume;
    SparseMatrix<double> Le, LeT;
    vector<SparseMatrix<double>> Le_list;
    Vector3d Force;
    MatrixXd due, due_sum;
    vector<VectorXd> f_sensor, x_sensor;
    vector<double> weight;
    double el, eu;
    //SimplicialLDLT<SparseMatrix<double>> solver;

    ElasticBody(double E, double lowerbound, double upperbound, int size, double nv);
    SparseMatrix<double> readSparseMatrixFromFile(const string& filename, int size); 
    VectorXd load_vector_from_file(const string& filename, int size);
    void load_data(int cnt, string filepath);
    void read_mesh(const string &nodes_filename, const string &cells_filename, vector<Vector3d> &nodes, vector<Vector4i> &cells);
    VectorXd gen_grad_f();
    SparseMatrix<double> gen_K();
    void gen_due();
    double eqn(VectorXd s);
    VectorXd jac(VectorXd s);
    
    // 设置当前节点位移 (用于更新 nodes_rt)
    void set_current_displacement(VectorXd u);
    
    // 设置单元杨氏模量分布 (用于非均匀场)
    void set_element_modulus(VectorXd E_vec);
    
    // 生成切线刚度矩阵 (基于当前 nodes_rt，不重置回 nodes0)
    SparseMatrix<double> gen_tangent_stiffness();

    // 计算单位几何力 (Unit Geometric Forces)
    // 返回值: MatrixXd (12行, num_cells列)
    // 每一列是一个单元对4个节点(12个自由度)产生的内力 (假设 E=1)
    MatrixXd gen_unit_geometric_forces();

private:
    vector<Vector3d> readNodes(const string &filename);
    vector<Vector4i> readCells(const string &filename);
    double tri_area(const vector<int> &idx, const MatrixXd &nodes);
};

#endif // ELASTICBODY_H
