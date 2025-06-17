#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <errno.h>
#include <time.h>

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
    int stability_test; // 是否进行稳定性测试
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

// 显式欧拉时间步进
void explicit_euler_step(double *u, double *u_old, int local_size, double dS, double dt, 
                         OptionParams *params, int rank, int size, MPI_Comm comm) {
    double alpha, beta, gamma;
    double S;
    
    // 复制旧值
    memcpy(u_old, u, local_size * sizeof(double));
    
    // 计算内部点
    for (int i = 1; i < local_size - 1; i++) {
        S = (rank * (params->N / size) + i) * dS;
        alpha = 0.5 * params->sigma * params->sigma * S * S;
        beta = params->r * S;
        gamma = params->r;
        
        double d2V_dS2 = (u_old[i+1] - 2*u_old[i] + u_old[i-1]) / (dS * dS);
        double dV_dS = (u_old[i+1] - u_old[i-1]) / (2 * dS);
        
        u[i] = u_old[i] + dt * (alpha * d2V_dS2 + beta * dV_dS - gamma * u_old[i]);
    }
    
    // 交换边界数据
    exchange_boundary_data(u, local_size, rank, size, comm);
    
    // 应用边界条件
    apply_boundary_condition(u, local_size, rank, size, params->current_time + dt, params);
}

// 三对角矩阵求解 (Thomas算法)
void solve_tridiagonal(double *a, double *b, double *c, double *d, double *x, int n) {
    // 前向消元
    for (int i = 1; i < n; i++) {
        double m = a[i] / b[i-1];
        b[i] = b[i] - m * c[i-1];
        d[i] = d[i] - m * d[i-1];
    }
    
    // 回代
    x[n-1] = d[n-1] / b[n-1];
    for (int i = n - 2; i >= 0; i--) {
        x[i] = (d[i] - c[i] * x[i+1]) / b[i];
    }
}

// 隐式欧拉时间步进
void implicit_euler_step(double *u, double *u_old, int local_size, double dS, double dt, 
                         OptionParams *params, int rank, int size, MPI_Comm comm) {
    double alpha, beta, gamma;
    double S;
    
    // 复制旧值
    memcpy(u_old, u, local_size * sizeof(double));
    
    // 分配三对角系统内存
    double *a = (double *)malloc(local_size * sizeof(double)); // 下对角线
    double *b = (double *)malloc(local_size * sizeof(double)); // 主对角线
    double *c = (double *)malloc(local_size * sizeof(double)); // 上对角线
    double *d = (double *)malloc(local_size * sizeof(double)); // 右侧向量
    
    // 构建线性系统
    for (int i = 0; i < local_size; i++) {
        S = (rank * (params->N / size) + i) * dS;
        alpha = 0.5 * params->sigma * params->sigma * S * S;
        beta = params->r * S;
        gamma = params->r;
        
        if (i == 0) {
            // 左边界
            a[i] = 0.0;
            b[i] = 1.0;
            c[i] = 0.0;
            d[i] = left_boundary_value(params->current_time + dt, params);
        } else if (i == local_size - 1) {
            // 右边界
            a[i] = 0.0;
            b[i] = 1.0;
            c[i] = 0.0;
            d[i] = right_boundary_value(params->current_time + dt, params);
        } else {
            // 内部点
            double coeff1 = alpha / (dS * dS) - beta / (2 * dS);
            double coeff2 = 1.0/dt + 2.0 * alpha / (dS * dS) + gamma;
            double coeff3 = alpha / (dS * dS) + beta / (2 * dS);
            
            a[i] = -dt * coeff1;
            b[i] = 1.0 + dt * coeff2;
            c[i] = -dt * coeff3;
            d[i] = u_old[i];
        }
    }
    
    // 求解三对角系统
    solve_tridiagonal(a, b, c, d, u, local_size);
    
    // 释放内存
    free(a);
    free(b);
    free(c);
    free(d);
    
    // 交换边界数据
    exchange_boundary_data(u, local_size, rank, size, comm);
    
    // 应用边界条件
    apply_boundary_condition(u, local_size, rank, size, params->current_time + dt, params);
}

