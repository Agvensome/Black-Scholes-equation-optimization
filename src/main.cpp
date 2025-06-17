#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <errno.h>

// 金融参数结构体
typedef struct {
    double K;           // 行权价
    double r;           // 无风险利率
    double sigma;       // 波动率
    double T;           // 到期时间
    double S_max;       // 资产价格上限
    int option_type;    // 0=看涨, 1=看跌
    int N;              // 空间网格数
    int time_steps;     // 时间步数
    int save_interval;  // 保存间隔
    char restart_file[256]; // 重启文件名
    double current_time; // 当前时间
} OptionParams;

// 求解器状态
typedef struct {
    double *u;          // 当前解向量
    double *u_old;      // 上一时间步解向量
    int current_step;    // 当前步数
} SolverState;

// 计算左边界值
double left_boundary_value(double t, OptionParams *params) {
    if (params->option_type == 1) { // 看跌期权
        return params->K * exp(-params->r * (params->T - t));
    }
    return 0.0; // 看涨期权
}

// 计算右边界值
double right_boundary_value(double t, OptionParams *params) {
    if (params->option_type == 0) { // 看涨期权
        return params->S_max - params->K * exp(-params->r * (params->T - t));
    }
    return 0.0; // 看跌期权
}

// 设置初始条件
void set_initial_condition(double *u, int local_size, int offset, OptionParams *params) {
    double dS = params->S_max / (params->N - 1);
    for (int i = 0; i < local_size; i++) {
        double S = (offset + i) * dS;
        if (params->option_type == 0) { // 看涨期权
            u[i] = (S > params->K) ? S - params->K : 0.0;
        } else { // 看跌期权
            u[i] = (S < params->K) ? params->K - S : 0.0;
        }
    }
}

// 应用边界条件
void apply_boundary_condition(double *u, int local_size, int rank, int size, 
                             double t, OptionParams *params) {
    // 左边界 (rank 0的第一个点)
    if (rank == 0) {
        u[0] = left_boundary_value(t, params);
    }
    
    // 右边界 (最后一个rank的最后一个点)
    if (rank == size - 1) {
        u[local_size - 1] = right_boundary_value(t, params);
    }
}

// 交换边界数据
void exchange_boundary_data(double *u, int local_size, int rank, int size, MPI_Comm comm) {
    double left_bound = 0.0, right_bound = 0.0;
    MPI_Status status;
    
    // 发送右边界给下一个进程，接收左边界从上一个进程
    if (rank < size - 1) {
        MPI_Send(&u[local_size - 1], 1, MPI_DOUBLE, rank + 1, 0, comm);
    }
    if (rank > 0) {
        MPI_Recv(&left_bound, 1, MPI_DOUBLE, rank - 1, 0, comm, &status);
        u[0] = left_bound;
    }
    
    // 发送左边界给上一个进程，接收右边界从下一个进程
    if (rank > 0) {
        MPI_Send(&u[0], 1, MPI_DOUBLE, rank - 1, 1, comm);
    }
    if (rank < size - 1) {
        MPI_Recv(&right_bound, 1, MPI_DOUBLE, rank + 1, 1, comm, &status);
        u[local_size - 1] = right_bound;
    }
}
