#include "ElasticBody.h"
ElasticBody::ElasticBody(double E, double lowerbound, double upperbound, int size, double nv)
{
        el = lowerbound;
        eu = upperbound;
        total_times = size;
        MatrixXd nodes_temp;
        read_mesh("nodes.txt", "cells.txt", nodes, cells);
        
        nodes0 = nodes;
        nodes_rt = nodes;
        num_nodes = nodes.size();cout<<num_nodes<<endl;
        num_cells = cells.size();cout<<num_cells<<endl;
        nodes_rt_list = vector<vector<Vector3d>>(total_times, vector<Vector3d>(num_nodes, Vector3d::Zero()));
        for (int i = 0; i<total_times; ++i){
            nodes_rt_list[i] = nodes;
        }
        //E = 4e6; 
        //nv = 0.49; 
        stepn = 100;
        //we = vector<double>(num_cells, E);
        we = E*VectorXd::Ones(num_cells);
        weight = vector<double> (total_times, 0.0);
        mu_o = 1.0 / (2.0 * (1.0 + nv));
        la_o = nv / ((1.0 + nv) * (1.0 - 2.0 * nv));
        
        Force = Vector3d(0,0,-10);
        volume = VectorXd(num_cells);
        f = VectorXd::Zero(3 * num_nodes);
        f_temp = VectorXd::Zero(3 * num_nodes);
        f_temp_list = vector<VectorXd> (num_cells);
        Dm_list = vector<Matrix3d>(num_cells);
        Dm_inv_list = vector<Matrix3d>(num_cells);
        due = MatrixXd::Zero(3*num_nodes, num_cells);
        

        dD[0] = (Matrix3d() << -1, -1, -1, 0, 0, 0, 0, 0, 0).finished();
        dD[1] = (Matrix3d() << 0, 0, 0, -1, -1, -1, 0, 0, 0).finished();
        dD[2] = (Matrix3d() << 0, 0, 0, 0, 0, 0, -1, -1, -1).finished();
        dD[3] = (Matrix3d() << 1, 0, 0, 0, 0, 0, 0, 0, 0).finished();
        dD[4] = (Matrix3d() << 0, 0, 0, 1, 0, 0, 0, 0, 0).finished();
        dD[5] = (Matrix3d() << 0, 0, 0, 0, 0, 0, 1, 0, 0).finished();
        dD[6] = (Matrix3d() << 0, 1, 0, 0, 0, 0, 0, 0, 0).finished();
        dD[7] = (Matrix3d() << 0, 0, 0, 0, 1, 0, 0, 0, 0).finished();
        dD[8] = (Matrix3d() << 0, 0, 0, 0, 0, 0, 0, 1, 0).finished();
        dD[9] = (Matrix3d() << 0, 0, 1, 0, 0, 0, 0, 0, 0).finished();
        dD[10] = (Matrix3d() << 0, 0, 0, 0, 0, 1, 0, 0, 0).finished();
        dD[11] = (Matrix3d() << 0, 0, 0, 0, 0, 0, 0, 0, 1).finished();


        Klocal = MatrixXd::Zero(12, 12);
        K = SparseMatrix<double>(3 * num_nodes, 3 * num_nodes);
        Le = SparseMatrix<double>(4 * 3 *num_cells, 3 * num_nodes);
        node_force = VectorXd::Zero(12*num_cells);
        node_force_t = VectorXd::Zero(12);
        Le_list = vector<SparseMatrix<double>>(num_cells, SparseMatrix<double>(12, 3*num_nodes));
        vector<Triplet<double>> triplet_list;
        f_sensor = vector<VectorXd>(total_times, VectorXd::Zero(num_nodes * 3));
        x_sensor = vector<VectorXd>(total_times, VectorXd::Zero(num_nodes * 3));

        int t=0;
        
        for (const auto &cell : cells) {
            vector<Triplet<double>> triplet_list_le;
            for (int i = 0; i < 4; ++i) {
                int n = cell[i];
                for(int j =0; j<3; ++j){
                    triplet_list_le.emplace_back(3*i+j, 3*n+j, 1);
                    triplet_list.emplace_back(3*i + 4*3*t + j, 3*n + j, 1);
                }
                node_pos0[i] = nodes0[n];
            }
            Le_list[t].setFromTriplets(triplet_list_le.begin(), triplet_list_le.end());
            Dm.col(0) = node_pos0[1] - node_pos0[0];
            Dm.col(1) = node_pos0[2] - node_pos0[0];
            Dm.col(2) = node_pos0[3] - node_pos0[0];
            Dm_list[t] = Dm;
            Dm_inv_list[t] = Dm.inverse();
            volume[t] = Dm.determinant() * 0.1666667;
            t+=1;
        }
        Le.setFromTriplets(triplet_list.begin(), triplet_list.end());
        LeT = Le.transpose();
        vector<Triplet<double>> A_triplets;
    
        A_list = vector<SparseMatrix<double>>(total_times);
        B_list = vector<SparseMatrix<double>>(total_times);
        //SparseMatrix<double> K_start = gen_K();
        //K_start.makeCompressed();
        //SimplicialLDLT<SparseMatrix<double>> solver;
        //solver.analyzePattern(K_start);
}
// 函数：从文件中读取稀疏矩阵
SparseMatrix<double> ElasticBody::readSparseMatrixFromFile(const string& filename, int size) {
    vector<Triplet<double>> triplets;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Unable to open file: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    int row, col;
    double value;
    while (file >> row >> col >> value) {
        triplets.emplace_back(row, col, value);
    }
    file.close();
    int rows = size;
    int cols = size;
    SparseMatrix<double> sparseMatrix(rows, cols);
    sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
    return sparseMatrix;
}
VectorXd ElasticBody::load_vector_from_file(const string& filename, int size) {
    ifstream file(filename);
    VectorXd vec(size);
    if (!file.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        exit(EXIT_FAILURE);
    }
    double value;
    for (int i = 0; i < size && file >> value; ++i) {
        vec(i) = value;
    }
    return vec;
}

void ElasticBody::load_data(int cnt, string filepath) {
    string pnt_filename;
    string force_filename;
    string matrixA_filename, matrixB_filename;
    for (int i = 0; i < total_times; ++i) {
        pnt_filename = filepath + "/pnt" + to_string(i + cnt) + ".txt";
        force_filename = filepath + "/force" + to_string(i + cnt) + ".txt";
        
        x_sensor[i] = load_vector_from_file(pnt_filename, 3 * num_nodes);
        f_sensor[i] = load_vector_from_file(force_filename, 3 * num_nodes);

        matrixA_filename = filepath + "/A-" + to_string(i + cnt) + ".txt";
        matrixB_filename = filepath + "/B-" + to_string(i + cnt) + ".txt";

        A_list[i] = readSparseMatrixFromFile(matrixA_filename, 3 * num_nodes);
        B_list[i] = readSparseMatrixFromFile(matrixB_filename, 3 * num_nodes);
      
       
    }
}

// 读取节点数据文件
vector<Vector3d> ElasticBody::readNodes(const string &filename) {
    vector<Vector3d> nodes;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开节点文件: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    Vector3d node;
    while (file >> node[0] >> node[1] >> node[2]) {
        nodes.push_back(node);
    }

    file.close();
    return nodes;
}

// 读取单元数据文件
vector<Vector4i> ElasticBody::readCells(const string &filename) {
    vector<Vector4i> cells;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "无法打开单元文件: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    Vector4i cell;
    while (file >> cell[0] >> cell[1] >> cell[2] >> cell[3]) {
        cells.push_back(cell);
    }

    file.close();
    return cells;
}

// 读取 mesh 数据的函数
void ElasticBody::read_mesh(const string &nodes_filename, const string &cells_filename, vector<Vector3d> &nodes, vector<Vector4i> &cells) {
    nodes = readNodes(nodes_filename);
    cells = readCells(cells_filename);
    
    cout << "Mesh data loaded: " << nodes_filename << " (" << nodes.size() << " nodes), " << cells_filename << " (" << cells.size() << " cells)" << endl;
}

double ElasticBody::tri_area(const vector<int> &idx, const MatrixXd &nodes) {
    // 计算三角形面积的辅助函数
    Vector3d p1 = nodes.row(idx[0]);
    Vector3d p2 = nodes.row(idx[1]);
    Vector3d p3 = nodes.row(idx[2]);
    return 0.5 * ((p2 - p1).cross(p3 - p1)).norm();
}




VectorXd ElasticBody::gen_grad_f(){
        
        int t = 0;
        for (const auto &cell : cells) {
            mu = we[t]*mu_o;
            la = we[t]*la_o;
            
            for (int i = 0; i < 4; ++i) {
                int n = cell[i];
                node_pos[i] = nodes_rt[n];
            }
            Ds.col(0) = node_pos[1] - node_pos[0];
            Ds.col(1) = node_pos[2] - node_pos[0];
            Ds.col(2) = node_pos[3] - node_pos[0];
            F = Ds * Dm_inv_list[t];
            Finv = F.inverse();
            FinvT = Finv.transpose();
            FtF = F.transpose()*F;
            J = F.determinant();
            logJ = log(J);
            Ic = FtF.trace();
            piola = mu * F - mu * FinvT + la * logJ * 0.5 * FinvT;
            H = -volume[t] * piola * Dm_inv_list[t].transpose();
            gradC.row(1) = H.col(0);
            gradC.row(2) = H.col(1);
            gradC.row(3) = H.col(2);
            gradC.row(0) = -H.col(0) - H.col(1) - H.col(2);
            for (int i = 0; i < 4; ++i){
                node_force.segment<3>(3*i + 3*4*t) = gradC.row(i);
            }
            ++t;
        }
        f = LeT*node_force;
        return f;
}
SparseMatrix<double> ElasticBody::gen_K(){
        vector<Triplet<double>> triplet_list;
        int t = 0;
        nodes_rt = nodes0;
        for (const auto &cell : cells) {
            
            mu = we[t]*mu_o;
            la = we[t]*la_o;
            
            for (int i = 0; i < 4; ++i) {
                int n = cell[i];
                node_pos[i] = nodes_rt[n];
            }
            Ds.col(0) = node_pos[1] - node_pos[0];
            Ds.col(1) = node_pos[2] - node_pos[0];
            Ds.col(2) = node_pos[3] - node_pos[0];
            F = Ds * Dm_inv_list[t];
            Finv = F.inverse();
            FinvT = Finv.transpose();
            FtF = F.transpose()*F;
            J = F.determinant();
            logJ = log(J);
            Ic = FtF.trace();
            for (int i = 0; i < 12; ++i) {
                dF[i] = dD[i] * Dm_inv_list[t];
                dP[i] = mu * dF[i] + (mu - 0.5* la * logJ) * (FinvT * dF[i].transpose() * FinvT);
                term = Finv * dF[i];
                dP[i] += la * (term(0, 0) + term(1, 1) + term(2, 2)) * FinvT;
                dH[i] = -volume[t] * dP[i] * Dm_inv_list[t].transpose();
                int j = i / 3;
                int d = i%3;
                triplet_list.emplace_back(3 * cell[1] , 3 * cell[j] + d, dH[i](0, 0));
                triplet_list.emplace_back(3 * cell[1] + 1, 3 * cell[j] + d, dH[i](1, 0));
                triplet_list.emplace_back(3 * cell[1] + 2, 3 * cell[j] + d, dH[i](2, 0));
                triplet_list.emplace_back(3 * cell[2], 3 * cell[j] + d, dH[i](0, 1));
                triplet_list.emplace_back(3 * cell[2] + 1, 3 * cell[j] + d, dH[i](1, 1));
                triplet_list.emplace_back(3 * cell[2] + 2, 3 * cell[j] + d, dH[i](2, 1));
                triplet_list.emplace_back(3 * cell[3] , 3 * cell[j] + d, dH[i](0, 2));
                triplet_list.emplace_back(3 * cell[3] + 1 , 3 * cell[j] + d, dH[i](1, 2));
                triplet_list.emplace_back(3 * cell[3] + 2 , 3 * cell[j] + d, dH[i](2, 2));
                triplet_list.emplace_back(3 * cell[0] , 3 * cell[j] + d, -dH[i](0, 0) - dH[i](0, 1) - dH[i](0, 2));
                triplet_list.emplace_back(3 * cell[0] + 1, 3 * cell[j] + d, -dH[i](1, 0) - dH[i](1, 1) - dH[i](1, 2));
                triplet_list.emplace_back(3 * cell[0] + 2, 3 * cell[j] + d, -dH[i](2, 0) - dH[i](2, 1) - dH[i](2, 2));
            }
            ++t;
        }
        K.setFromTriplets(triplet_list.begin(), triplet_list.end());
        return K;
}


void ElasticBody::gen_due(){
        vector<Triplet<double>> triplet_list2;
        int t = 0;
        for (const auto &cell : cells) {
            for (int i = 0; i < 4; ++i) {
                int n = cell[i];
                node_pos[i] = nodes_rt[n];
            }
            
            Ds.col(0) = node_pos[1] - node_pos[0];
            Ds.col(1) = node_pos[2] - node_pos[0];
            Ds.col(2) = node_pos[3] - node_pos[0];
            F = Ds * Dm_inv_list[t];
            Finv = F.inverse();
            FinvT = Finv.transpose();
            FtF = F.transpose()*F;
            J = F.determinant();
            logJ = log(J);
            
            //Ic = FtF.trace();
            piola = mu_o*F - mu_o* FinvT + la_o * logJ * 0.5 * FinvT;
            H = -volume[t] * piola * Dm_inv_list[t].transpose();
            gradC.row(1) = H.col(0);
            gradC.row(2) = H.col(1);
            gradC.row(3) = H.col(2);
            gradC.row(0) = -H.col(0) - H.col(1) - H.col(2);
            
            for (int i = 0; i < 4; ++i){
                node_force_t.segment<3>(3*i) = gradC.row(i);
            }
            f_temp_list[t] = Le_list[t].transpose() * node_force_t;
            mu = we[t]*mu_o;
            la = we[t]*la_o;
            for (int i = 0; i < 12; ++i) {
                dF[i] = dD[i] * Dm_inv_list[t];
                dP[i] = mu * dF[i] + (mu - 0.5 * la * logJ) * (FinvT * dF[i].transpose() * FinvT);
                term = Finv * dF[i];
                dP[i] += la * (term(0, 0) + term(1, 1) + term(2, 2)) * FinvT;
                dH[i] = -volume[t] * dP[i] * Dm_inv_list[t].transpose();
                int j = i / 3;
                int d = i%3;
                triplet_list2.emplace_back(3 * cell[1] , 3 * cell[j] + d, dH[i](0, 0));
                triplet_list2.emplace_back(3 * cell[1] + 1, 3 * cell[j] + d, dH[i](1, 0));
                triplet_list2.emplace_back(3 * cell[1] + 2, 3 * cell[j] + d, dH[i](2, 0));
                triplet_list2.emplace_back(3 * cell[2], 3 * cell[j] + d, dH[i](0, 1));
                triplet_list2.emplace_back(3 * cell[2] + 1, 3 * cell[j] + d, dH[i](1, 1));
                triplet_list2.emplace_back(3 * cell[2] + 2, 3 * cell[j] + d, dH[i](2, 1));
                triplet_list2.emplace_back(3 * cell[3] , 3 * cell[j] + d, dH[i](0, 2));
                triplet_list2.emplace_back(3 * cell[3] + 1 , 3 * cell[j] + d, dH[i](1, 2));
                triplet_list2.emplace_back(3 * cell[3] + 2 , 3 * cell[j] + d, dH[i](2, 2));
                triplet_list2.emplace_back(3 * cell[0] , 3 * cell[j] + d, -dH[i](0, 0) - dH[i](0, 1) - dH[i](0, 2));
                triplet_list2.emplace_back(3 * cell[0] + 1, 3 * cell[j] + d, -dH[i](1, 0) - dH[i](1, 1) - dH[i](1, 2));
                triplet_list2.emplace_back(3 * cell[0] + 2, 3 * cell[j] + d, -dH[i](2, 0) - dH[i](2, 1) - dH[i](2, 2));
            }
            ++t;
        }
        K.setFromTriplets(triplet_list2.begin(), triplet_list2.end());
        //Kw.setFromTriplets(triplet_list1.begin(), triplet_list1.end());
        K -= A;
        
        K.makeCompressed();
        //cout<<K<<endl;
        SimplicialLDLT<SparseMatrix<double>> solver;
        solver.analyzePattern(K);
        solver.factorize(K);

        #pragma omp parallel for
        for (int t = 0; t < num_cells; ++t) {
            auto temp_result = solver.solve(f_temp_list[t]);
            due.col(t) = temp_result;
            //cout<<due.col(t).transpose()<<endl;
        }
        //cout<<f_temp_list[0].transpose()<<endl;
        return;
}

double ElasticBody::eqn(VectorXd s){
    double out = 0.0;
    we = el*s + eu*(VectorXd::Ones(num_cells) - s);
    
    SparseMatrix<double> K_start = gen_K();
    K_start.makeCompressed();
    double damping_factor = 1;
    for(int k = 0; k<total_times; ++k){
        VectorXd dd = VectorXd::Zero(num_nodes * 3);
        SimplicialLDLT<SparseMatrix<double>> solver_forward;
        K = K_start - A_list[k];
        K.makeCompressed();
        solver_forward.compute(K);
        VectorXd f_inter;
        VectorXd dP;
        VectorXd nodes_flatterned = Map<VectorXd>(nodes0.data()->data(), 3 * num_nodes);
        VectorXd nodes_flatterned0 = Map<VectorXd>(nodes0.data()->data(), 3 * num_nodes);

        for (int i = 0; i < stepn; ++i) {
            for (size_t j = 0; j < num_nodes; ++j) {
                nodes_rt[j] = nodes_flatterned.segment<3>(3 * j);
            }
            //K = gen_K() - A_list[k];
            f_inter = gen_grad_f();
            dP = -(f_sensor[k] + f_inter - A_list[k] * (nodes_flatterned - nodes_flatterned0));
            dd = solver_forward.solve(dP);
            //d += dd;
            nodes_flatterned += damping_factor*dd;
            //cout<<dP.norm()<<endl;
            //if (dP.norm() < 0.001 * f_sensor[k].cwiseAbs().maxCoeff()) {
            if (dP.norm() < 1e-4) {

                break;
            }
            
        }
        
        for (size_t j = 0; j < num_nodes; ++j) {
            nodes_rt_list[k][j] = nodes_flatterned.segment<3>(3 * j);
        }
        VectorXd diff = B_list[k]*(nodes_flatterned - x_sensor[k]);
        double value = diff.transpose()*diff;
        out += weight[k]*value;
        out += value;
        
    }
    return 1e12*out;
}


VectorXd ElasticBody::jac(VectorXd s){
    double out = 0.0;
    we = el*s + eu*(VectorXd::Ones(num_cells) - s);
    VectorXd doe = VectorXd::Zero(num_cells);
    due_sum = MatrixXd::Zero(3*num_nodes, num_cells);
    for (int k = 0; k<total_times; ++k){
        vector<Triplet<double>> triplet_list;
        VectorXd nodes_flatterned = Map<VectorXd>(nodes_rt_list[k].data()->data(), 3 * num_nodes);
        VectorXd diff = nodes_flatterned - x_sensor[k];
        VectorXd dou = B_list[k].transpose()*B_list[k]*diff;
        nodes_rt = nodes_rt_list[k];
        A = A_list[k];
        gen_due();
        //doe += weight[k]*dou.transpose()*due;
        doe += dou.transpose()*due;
        due_sum += B_list[k]*due;

    }
    return -2e12*(el- eu)*doe;
}

// ==========================================
// [新增] Phase 1: 前向仿真接口实现
// ==========================================

void ElasticBody::set_current_displacement(VectorXd u) {
    // 检查维度
    if (u.size() != 3 * num_nodes) {
        cerr << "Error: Displacement vector size mismatch." << endl;
        return;
    }
    
    // 将扁平的位移向量映射到 nodes_rt (Current Configuration)
    // x_current = x_initial + u
    for (size_t i = 0; i < num_nodes; ++i) {
        nodes_rt[i] = nodes0[i] + u.segment<3>(3 * i);
    }
}

void ElasticBody::set_element_modulus(VectorXd E_vec) {
    if (E_vec.size() != num_cells) {
        cerr << "Error: Young's modulus vector size mismatch. Expected " << num_cells << endl;
        return;
    }
    // 更新内部的单元刚度向量
    we = E_vec;
}

SparseMatrix<double> ElasticBody::gen_tangent_stiffness() {
    // 逻辑与 gen_K 完全一致，唯一的区别是：
    // 我们移除了 "nodes_rt = nodes0;" 这一行。
    // 这样矩阵就会基于 set_current_displacement 设置的变形后构型进行计算。

    vector<Triplet<double>> triplet_list;
    int t = 0;
    
    // [CRITICAL DIFFERENCE]: 不要重置 nodes_rt !
    // nodes_rt = nodes0; <--- REMOVED
    
    for (const auto &cell : cells) {
        
        mu = we[t]*mu_o;
        la = we[t]*la_o;
        
        for (int i = 0; i < 4; ++i) {
            int n = cell[i];
            node_pos[i] = nodes_rt[n]; // 这里使用的是当前的 deformed nodes
        }
        
        // 下面的物理计算逻辑与 gen_K 保持完全一致
        Ds.col(0) = node_pos[1] - node_pos[0];
        Ds.col(1) = node_pos[2] - node_pos[0];
        Ds.col(2) = node_pos[3] - node_pos[0];
        F = Ds * Dm_inv_list[t];
        Finv = F.inverse();
        FinvT = Finv.transpose();
        J = F.determinant();
        logJ = log(J);

        // 如果 J 非正，可能会导致 logJ 崩溃，建议在这里加一个简单的保护或警告
        // 但为了保持与 gen_K 逻辑一致，此处暂不修改
        
        for (int i = 0; i < 12; ++i) {
            dF[i] = dD[i] * Dm_inv_list[t];
            dP[i] = mu * dF[i] + (mu - 0.5* la * logJ) * (FinvT * dF[i].transpose() * FinvT);
            term = Finv * dF[i];
            dP[i] += la * (term(0, 0) + term(1, 1) + term(2, 2)) * FinvT;
            dH[i] = -volume[t] * dP[i] * Dm_inv_list[t].transpose();
            
            // 组装 Triplet
            int j = i / 3;
            int d = i % 3;
            triplet_list.emplace_back(3 * cell[1] , 3 * cell[j] + d, dH[i](0, 0));
            triplet_list.emplace_back(3 * cell[1] + 1, 3 * cell[j] + d, dH[i](1, 0));
            triplet_list.emplace_back(3 * cell[1] + 2, 3 * cell[j] + d, dH[i](2, 0));
            triplet_list.emplace_back(3 * cell[2], 3 * cell[j] + d, dH[i](0, 1));
            triplet_list.emplace_back(3 * cell[2] + 1, 3 * cell[j] + d, dH[i](1, 1));
            triplet_list.emplace_back(3 * cell[2] + 2, 3 * cell[j] + d, dH[i](2, 1));
            triplet_list.emplace_back(3 * cell[3] , 3 * cell[j] + d, dH[i](0, 2));
            triplet_list.emplace_back(3 * cell[3] + 1 , 3 * cell[j] + d, dH[i](1, 2));
            triplet_list.emplace_back(3 * cell[3] + 2 , 3 * cell[j] + d, dH[i](2, 2));
            triplet_list.emplace_back(3 * cell[0] , 3 * cell[j] + d, -dH[i](0, 0) - dH[i](0, 1) - dH[i](0, 2));
            triplet_list.emplace_back(3 * cell[0] + 1, 3 * cell[j] + d, -dH[i](1, 0) - dH[i](1, 1) - dH[i](1, 2));
            triplet_list.emplace_back(3 * cell[0] + 2, 3 * cell[j] + d, -dH[i](2, 0) - dH[i](2, 1) - dH[i](2, 2));
        }
        ++t;
    }
    
    // 构建稀疏矩阵
    SparseMatrix<double> K_tangent(3 * num_nodes, 3 * num_nodes);
    K_tangent.setFromTriplets(triplet_list.begin(), triplet_list.end());
    return K_tangent;
}

MatrixXd ElasticBody::gen_unit_geometric_forces() {
    // 这是一个 12 x Num_Cells 的稠密矩阵
    // 列索引 t 对应第 t 个单元
    // 行索引 0-11 对应 4个节点 x 3个自由度 (x0, y0, z0, x1, y1, z1, ...)
    MatrixXd unit_forces(12, num_cells);
    
    int t = 0;
    
    // 注意：这里我们基于当前的 nodes_rt (Current Configuration) 进行计算
    // 必须确保 Python 端在调用此函数前，已经调用过 set_current_displacement 更新了 nodes_rt
    
    for (const auto &cell : cells) {
        
        // 1. 获取当前变形后的节点坐标
        for (int i = 0; i < 4; ++i) {
            node_pos[i] = nodes_rt[cell[i]];
        }
        
        // 2. 计算变形梯度 F = Ds * Dm_inv
        Ds.col(0) = node_pos[1] - node_pos[0];
        Ds.col(1) = node_pos[2] - node_pos[0];
        Ds.col(2) = node_pos[3] - node_pos[0];
        F = Ds * Dm_inv_list[t];
        
        // 3. 计算运动学量
        Finv = F.inverse();
        FinvT = Finv.transpose();
        J = F.determinant();
        logJ = log(J); // 注意：如果单元翻转(J<=0)，这里会出错。由外部保证网格质量。
        
        // 4. 计算单位第一类 Piola-Kirchhoff 应力 (Unit P)
        // 假设 E=1，则 mu = mu_o, la = la_o
        // 公式与 gen_grad_f 保持一致，只是去掉了 we[t]
        piola = mu_o * F - mu_o * FinvT + la_o * logJ * 0.5 * FinvT;
        
        // 5. 计算单位节点力矩阵 H (Unit Forces)
        // H 的每一列对应一个节点 (node1, node2, node3) 的力贡献
        // node0 的力由平衡条件得出
        H = -volume[t] * piola * Dm_inv_list[t].transpose();
        
        // 6. 展开为 12x1 向量并存入矩阵
        // H 的列: [f_node1, f_node2, f_node3]
        // f_node0 = - (f_node1 + f_node2 + f_node3)
        
        Vector3d f1 = H.col(0);
        Vector3d f2 = H.col(1);
        Vector3d f3 = H.col(2);
        Vector3d f0 = -f1 - f2 - f3;
        
        // 存入列 t
        unit_forces.col(t) << f0, f1, f2, f3;
        
        ++t;
    }
    
    return unit_forces;
}