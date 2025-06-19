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

// 后处理结果结构体
typedef struct {
    double S;         // 资产价格
    double price;     // 期权价格
    double delta;     // Delta值 (?V/?S)
} PostResult;

// 函数声明
int compute_option_price(OptionParams *params, int use_implicit, MPI_Comm comm);
int compute_option_price_verification(OptionParams *params, int use_implicit, MPI_Comm comm,
                                     double **u_final, int *local_size_final, int *offset_final);

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

// 保存重启文件
void save_restart_file(SolverState *state, int local_size, int offset, 
                       OptionParams *params, int rank) {
    // 如果没有指定重启文件名，则不保存
    if (strlen(params->restart_file) == 0) return;
    
    char filename[256];
    snprintf(filename, sizeof(filename), "%s_%d.dat", params->restart_file, rank);
    
    FILE *fp = fopen(filename, "wb");
    if (fp) {
        // 保存元数据
        fwrite(&state->current_step, sizeof(int), 1, fp);
        fwrite(&params->current_time, sizeof(double), 1, fp);
        fwrite(&offset, sizeof(int), 1, fp);
        fwrite(&local_size, sizeof(int), 1, fp);
        
        // 保存解向量
        fwrite(state->u, sizeof(double), local_size, fp);
        
        fclose(fp);
        if (rank == 0 && !params->stability_test) {
            printf("Saved restart file: %s\n", filename);
        }
    } else {
        if (rank == 0 && !params->stability_test) {
            perror("Failed to save restart file");
        }
    }
}

// 加载重启文件
int load_restart_file(SolverState *state, int *local_size, int *offset, 
                      OptionParams *params, int rank) {
    // 如果没有指定重启文件名，则不加载
    if (strlen(params->restart_file) == 0) return 0;
    
    char filename[256];
    snprintf(filename, sizeof(filename), "%s_%d.dat", params->restart_file, rank);
    
    FILE *fp = fopen(filename, "rb");
    if (fp) {
        // 读取元数据
        fread(&state->current_step, sizeof(int), 1, fp);
        fread(&params->current_time, sizeof(double), 1, fp);
        fread(offset, sizeof(int), 1, fp);
        fread(local_size, sizeof(int), 1, fp);
        
        // 分配内存
        state->u = (double *)malloc(*local_size * sizeof(double));
        state->u_old = (double *)malloc(*local_size * sizeof(double));
        
        // 读取解向量
        fread(state->u, sizeof(double), *local_size, fp);
        memcpy(state->u_old, state->u, *local_size * sizeof(double));
        
        fclose(fp);
        if (rank == 0 && !params->stability_test) {
            printf("Loaded restart file: %s\n", filename);
        }
        return 1; // 成功加载
    } else {
        if (rank == 0 && !params->stability_test) {
            printf("Restart file not found: %s. Starting from initial conditions.\n", filename);
        }
        return 0; // 文件未找到
    }
}

// 检查数值解是否发散
int check_divergence(double *u, int local_size, MPI_Comm comm) {
    int local_diverged = 0;
    int global_diverged = 0;
    
    // 检查本进程数据是否发散
    for (int i = 0; i < local_size; i++) {
        if (isnan(u[i]) || isinf(u[i]) || fabs(u[i]) > 1e100) {
            local_diverged = 1;
            break;
        }
    }
    
    // 全局检查是否发散
    MPI_Allreduce(&local_diverged, &global_diverged, 1, MPI_INT, MPI_LOR, comm);
    
    return global_diverged;
}

// 计算期权价格
int compute_option_price(OptionParams *params, int use_implicit, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // 计算每个进程的局部网格大小
    int local_size = params->N / size;
    int remainder = params->N % size;
    
    // 处理不能整除的情况
    if (rank < remainder) {
        local_size++;
    }
    
    int offset = rank * (params->N / size);
    if (rank < remainder) {
        offset += rank;
    } else {
        offset += remainder;
    }
    
    double dS = params->S_max / (params->N - 1);
    double dt = params->T / params->time_steps;
    
    // 初始化求解器状态
    SolverState state;
    state.current_step = 0;
    params->current_time = 0.0; // 初始化当前时间
    
    // 检查是否需要重启
    int restart_loaded = 0;
    if (strlen(params->restart_file)) {
        restart_loaded = load_restart_file(&state, &local_size, &offset, params, rank);
    }
    
    if (!restart_loaded) {
        // 分配内存
        state.u = (double *)malloc(local_size * sizeof(double));
        state.u_old = (double *)malloc(local_size * sizeof(double));
        
        // 设置初始条件
        set_initial_condition(state.u, local_size, offset, params);
        memcpy(state.u_old, state.u, local_size * sizeof(double));
        
        // 应用初始边界条件
        apply_boundary_condition(state.u, local_size, rank, size, 0.0, params);
    }
    
    // 时间步进循环
    int diverged = 0;
    for (int step = 0; step < params->time_steps; step++) {
        if (diverged) break;
        
        if (use_implicit) {
            implicit_euler_step(state.u, state.u_old, local_size, dS, dt, params, rank, size, comm);
        } else {
            explicit_euler_step(state.u, state.u_old, local_size, dS, dt, params, rank, size, comm);
        }
        
        // 检查是否发散
        if (step % 10 == 0) {
            diverged = check_divergence(state.u, local_size, comm);
            if (diverged) {
                if (rank == 0) {
                    printf("计算在时间步 %d (时间 %.4f) 发散\n", state.current_step, params->current_time);
                }
                break;
            }
        }
        
        state.current_step++;
        params->current_time += dt;
        
        // 定期保存重启文件
        if (params->save_interval > 0 && state.current_step % params->save_interval == 0) {
            save_restart_file(&state, local_size, offset, params, rank);
        }
    }
    
    // 保存最终结果
    if (params->save_interval > 0 && strlen(params->restart_file) && !diverged) {
        save_restart_file(&state, local_size, offset, params, rank);
    }
    
    // 清理内存
    free(state.u);
    free(state.u_old);
    
    return diverged ? 1 : 0;
}

// 计算期权价格 (用于验证)
int compute_option_price_verification(OptionParams *params, int use_implicit, MPI_Comm comm,
                                     double **u_final, int *local_size_final, int *offset_final) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // 计算每个进程的局部网格大小
    int local_size = params->N / size;
    int remainder = params->N % size;
    
    // 处理不能整除的情况
    if (rank < remainder) {
        local_size++;
    }
    
    int offset = rank * (params->N / size);
    if (rank < remainder) {
        offset += rank;
    } else {
        offset += remainder;
    }
    
    double dS = params->S_max / (params->N - 1);
    double dt = params->T / params->time_steps;
    
    // 初始化求解器状态
    SolverState state;
    state.current_step = 0;
    params->current_time = 0.0; // 初始化当前时间
    
    // 检查是否需要重启
    int restart_loaded = 0;
    if (strlen(params->restart_file)) {
        restart_loaded = load_restart_file(&state, &local_size, &offset, params, rank);
    }
    
    if (!restart_loaded) {
        // 分配内存
        state.u = (double *)malloc(local_size * sizeof(double));
        state.u_old = (double *)malloc(local_size * sizeof(double));
        
        // 设置初始条件
        set_initial_condition(state.u, local_size, offset, params);
        memcpy(state.u_old, state.u, local_size * sizeof(double));
        
        // 应用初始边界条件
        apply_boundary_condition(state.u, local_size, rank, size, 0.0, params);
    }
    
    // 时间步进循环
    int diverged = 0;
    for (int step = 0; step < params->time_steps; step++) {
        if (diverged) break;
        
        if (use_implicit) {
            implicit_euler_step(state.u, state.u_old, local_size, dS, dt, params, rank, size, comm);
        } else {
            explicit_euler_step(state.u, state.u_old, local_size, dS, dt, params, rank, size, comm);
        }
        
        // 检查是否发散
        if (step % 10 == 0) {
            diverged = check_divergence(state.u, local_size, comm);
            if (diverged) {
                if (rank == 0) {
                    printf("计算在时间步 %d (时间 %.4f) 发散\n", state.current_step, params->current_time);
                }
                break;
            }
        }
        
        state.current_step++;
        params->current_time += dt;
        
        // 定期保存重启文件
        if (params->save_interval > 0 && state.current_step % params->save_interval == 0) {
            save_restart_file(&state, local_size, offset, params, rank);
        }
    }
    
    // 保存最终结果
    if (params->save_interval > 0 && strlen(params->restart_file) && !diverged) {
        save_restart_file(&state, local_size, offset, params, rank);
    }
    
    // 处理最终结果
    if (!diverged) {
        *u_final = state.u;
        *local_size_final = local_size;
        *offset_final = offset;
        free(state.u_old);  // 只释放u_old，保留u_final
    } else {
        // 释放内存
        free(state.u);
        free(state.u_old);
    }
    
    return diverged ? 1 : 0;
}

// ======================== 后处理功能 ========================

// 保存后处理结果到文件
void save_post_results(PostResult *results, int count, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp) {
        fprintf(fp, "# Asset Price (S) | Option Price | Delta (dV/dS)\n");
        for (int i = 0; i < count; i++) {
            fprintf(fp, "%.6f %.6f %.6f\n", 
                   results[i].S, results[i].price, results[i].delta);
        }
        fclose(fp);
        printf("Saved post-processing results to: %s\n", filename);
    } else {
        perror("Failed to save post-processing results");
    }
}

// 计算Delta值 (?V/?S)
void compute_delta(double *u, int n, double dS, double *delta) {
    // 左边界：前向差分
    delta[0] = (u[1] - u[0]) / dS;
    
    // 内部点：中心差分
    for (int i = 1; i < n - 1; i++) {
        delta[i] = (u[i+1] - u[i-1]) / (2 * dS);
    }
    
    // 右边界：后向差分
    delta[n-1] = (u[n-1] - u[n-2]) / dS;
}

// 运行后处理任务
void run_postprocessing(OptionParams *params, int use_implicit, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    if (rank == 0) {
        printf("\n===== Running Postprocessing =====\n");
        printf("Option type: %s\n", params->option_type ? "Put" : "Call");
        printf("Strike price (K): %.2f\n", params->K);
        printf("Risk-free rate (r): %.4f\n", params->r);
        printf("Volatility (sigma): %.4f\n", params->sigma);
        printf("Maturity (T): %.2f\n", params->T);
        printf("Asset max (S_max): %.2f\n", params->S_max);
        printf("Space points (N): %d\n", params->N);
        printf("Time steps: %d\n", params->time_steps);
        printf("Method: %s\n", use_implicit ? "Implicit Euler" : "Explicit Euler");
    }
    
    // 计算空间步长
    double dS = params->S_max / (params->N - 1);
    
    // 计算时间步长
    double dt = params->T / params->time_steps;
    
    // 输出时间点
    double output_times[] = {
        0.0,               // t = 0
        0.25 * params->T,  // t = 0.25T
        0.50 * params->T,  // t = 0.50T
        0.75 * params->T,  // t = 0.75T
        params->T          // t = T
    };
    int num_output_times = sizeof(output_times) / sizeof(output_times[0]);
    
    // 每个进程的局部网格大小
    int local_size = params->N / size;
    int remainder = params->N % size;
    if (rank < remainder) local_size++;
    
    // 偏移量
    int offset = rank * (params->N / size);
    if (rank < remainder) {
        offset += rank;
    } else {
        offset += remainder;
    }
    
    // 初始化求解器状态
    SolverState state;
    state.current_step = 0;
    params->current_time = 0.0;
    
    // 分配内存
    state.u = (double *)malloc(local_size * sizeof(double));
    state.u_old = (double *)malloc(local_size * sizeof(double));
    
    // 设置初始条件
    set_initial_condition(state.u, local_size, offset, params);
    memcpy(state.u_old, state.u, local_size * sizeof(double));
    
    // 应用初始边界条件
    apply_boundary_condition(state.u, local_size, rank, size, 0.0, params);
    
    // 收集全局解到主进程
    double *global_u = NULL;
    if (rank == 0) {
        global_u = (double *)malloc(params->N * sizeof(double));
    }
    
    // 收集每个进程的局部大小
    int *recvcounts = NULL;
    int *displs = NULL;
    if (rank == 0) {
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
    }
    
    // 收集局部大小
    MPI_Gather(&local_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, comm);
    
    // 计算位移
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    // 主进程存储结果
    PostResult *results = NULL;
    if (rank == 0) {
        results = (PostResult *)malloc(params->N * sizeof(PostResult));
    }
    
    // 时间步进循环
    for (int step = 0; step < params->time_steps; step++) {
        // 选择时间步进方法
        if (use_implicit) {
            implicit_euler_step(state.u, state.u_old, local_size, dS, dt, 
                               params, rank, size, comm);
        } else {
            explicit_euler_step(state.u, state.u_old, local_size, dS, dt, 
                               params, rank, size, comm);
        }
        
        state.current_step++;
        params->current_time += dt;
        
        // 检查是否需要输出
        for (int i = 0; i < num_output_times; i++) {
            double time_diff = fabs(params->current_time - output_times[i]);
            if (time_diff < 0.5 * dt) {
                // 收集全局解
                MPI_Gatherv(state.u, local_size, MPI_DOUBLE,
                           global_u, recvcounts, displs, MPI_DOUBLE,
                           0, comm);
                
                if (rank == 0) {
                    // 计算Delta值
                    double *delta = (double *)malloc(params->N * sizeof(double));
                    compute_delta(global_u, params->N, dS, delta);
                    
                    // 准备结果
                    for (int j = 0; j < params->N; j++) {
                        results[j].S = j * dS;
                        results[j].price = global_u[j];
                        results[j].delta = delta[j];
                    }
                    
                    // 保存结果为.dat文件
                    char filename[256];
                    const char *method_name = use_implicit ? "implicit" : "explicit";
                    snprintf(filename, sizeof(filename), 
                             "results_t%.2f_%s.dat", output_times[i], method_name);
                    save_post_results(results, params->N, filename);
                    
                    free(delta);
                }
            }
        }
    }
    
    // 保存最终结果
    MPI_Gatherv(state.u, local_size, MPI_DOUBLE,
               global_u, recvcounts, displs, MPI_DOUBLE,
               0, comm);
    
    if (rank == 0) {
        // 计算Delta值
        double *delta = (double *)malloc(params->N * sizeof(double));
        compute_delta(global_u, params->N, dS, delta);
        
        // 准备结果
        for (int j = 0; j < params->N; j++) {
            results[j].S = j * dS;
            results[j].price = global_u[j];
            results[j].delta = delta[j];
        }
        
        // 保存结果为.dat文件
        char filename[256];
        const char *method_name = use_implicit ? "implicit" : "explicit";
        snprintf(filename, sizeof(filename), "results_final_%s.dat", method_name);
        save_post_results(results, params->N, filename);
        
        free(delta);
    }
    
    // 清理资源
    free(state.u);
    free(state.u_old);
    if (rank == 0) {
        free(global_u);
        free(recvcounts);
        free(displs);
        free(results);
    }
}

// ======================== 数值稳定性测试 ========================

// 数值稳定性测试函数
void numerical_stability_test(OptionParams *params, int use_implicit, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    // 减少空间网格点数以增大 ΔS
    const int fixed_N = 100;
    params->N = fixed_N;
    
    double dS = params->S_max / (fixed_N - 1);
    double stability_bound = (dS * dS) / (params->sigma * params->sigma * params->S_max * params->S_max);
    
    // 主进程创建输出文件
    FILE *fp = NULL;
    if (rank == 0) {
        char filename[256];
        snprintf(filename, sizeof(filename), "stability_test_%s.dat", 
                use_implicit ? "implicit" : "explicit");
        
        fp = fopen(filename, "w");
        if (fp) {
            fprintf(fp, "# Numerical Stability Test Results\n");
            fprintf(fp, "# Method: %s\n", use_implicit ? "Implicit Euler" : "Explicit Euler");
            fprintf(fp, "# Stability Bound: %.6f\n", stability_bound);
            fprintf(fp, "# Columns: TimeSteps DeltaT DeltaT_Ratio Stability(0=stable,1=diverged)\n");
            fprintf(fp, "# ----------------------------------------------------------------\n");
        } else {
            perror("Failed to create stability test output file");
        }
        
        printf("数值稳定性测试 - %s方法\n", use_implicit ? "隐式" : "显式");
        printf("空间网格点数 N = %d\n", fixed_N);
        printf("空间步长 ΔS = %.6f\n", dS);
        printf("理论稳定性边界 Δt < %.6f\n", stability_bound);
        printf("时间步数 | 时间步长 Δt | Δt/边界 | 稳定性\n");
        printf("-------------------------------------------\n");
    }
    
    // 测试参数设置
    const int num_tests = 15;
    double time_step_ratios[num_tests];
    double min_ratio = 0.1;
    double max_ratio = use_implicit ? 10.0 : 3.0;
    double step = (max_ratio - min_ratio) / (num_tests - 1);
    
    for (int i = 0; i < num_tests; i++) {
        time_step_ratios[i] = min_ratio + i * step;
    }
    
    // 执行测试
    for (int i = 0; i < num_tests; i++) {
        double ratio = time_step_ratios[i];
        double dt = ratio * stability_bound;
        params->time_steps = (int)ceil(params->T / dt);
        if (params->time_steps < 1) params->time_steps = 1;
        
        // 使用原始函数进行稳定性测试
        int result = compute_option_price(params, use_implicit, comm);
        int diverged;
        MPI_Reduce(&result, &diverged, 1, MPI_INT, MPI_MAX, 0, comm);
        
        // 主进程输出结果
        if (rank == 0) {
            printf("%6d | %10.6f | %7.2f | %s\n", 
                   params->time_steps, dt, ratio, 
                   diverged ? "发散" : "稳定");
            
            // 写入文件
            if (fp) {
                fprintf(fp, "%d %.6f %.6f %d\n", 
                        params->time_steps, dt, ratio, diverged);
                fflush(fp);  // 确保每次写入后刷新
            }
            
            fflush(stdout);
        }
    }
    
    // 清理资源
    if (rank == 0) {
        printf("-------------------------------------------\n");
        printf("测试完成。结果已保存至: stability_test_%s.dat\n", 
              use_implicit ? "implicit" : "explicit");
        
        if (fp) {
            fclose(fp);
        }
    }
}

// ======================== 代码验证功能 ========================

// 标准正态分布的累积分布函数 (CDF)
double norm_cdf(double x) {
    return 0.5 * (1 + erf(x / sqrt(2.0)));
}

// 计算Black-Scholes解析解
double black_scholes_exact(double S, double t, OptionParams *params) {
    double T = params->T;
    double tau = T - t;
    if (tau <= 0) {  // 到期时刻
        if (params->option_type == 0) { // 看涨期权
            return fmax(S - params->K, 0.0);
        } else { // 看跌期权
            return fmax(params->K - S, 0.0);
        }
    }
    
    double d1 = (log(S / params->K) + (params->r + 0.5 * params->sigma * params->sigma) * tau) 
                / (params->sigma * sqrt(tau));
    double d2 = d1 - params->sigma * sqrt(tau);
    
    if (params->option_type == 0) { // 看涨期权
        return S * norm_cdf(d1) - params->K * exp(-params->r * tau) * norm_cdf(d2);
    } else { // 看跌期权
        return params->K * exp(-params->r * tau) * norm_cdf(-d2) - S * norm_cdf(-d1);
    }
}

// 计算最大模误差
double calculate_max_error(double *u_num, int local_size, int offset, 
                          OptionParams *params, double dS, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    double local_max_error = 0.0;
    for (int i = 0; i < local_size; i++) {
        double S = (offset + i) * dS;
        double u_exact = black_scholes_exact(S, 0.0, params); // t=0时刻
        double error = fabs(u_exact - u_num[i]);
        if (error > local_max_error) {
            local_max_error = error;
        }
    }
    
    double global_max_error;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    return global_max_error;
}

// 空间收敛性测试 (输出为.dat文件)
void spatial_convergence_test(OptionParams *params, int use_implicit, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    // 创建输出文件
    FILE *out_file = NULL;
    if (rank == 0) {
        char out_filename[256];
        snprintf(out_filename, sizeof(out_filename), "spatial_convergence_%s.dat", 
                use_implicit ? "implicit" : "explicit");
        
        out_file = fopen(out_filename, "w");
        if (!out_file) {
            perror("Failed to create spatial convergence output file");
            return;
        }
        
        // 写入文件头
        fprintf(out_file, "# Spatial Convergence Test Results\n");
        fprintf(out_file, "# Method: %s\n", use_implicit ? "Implicit Euler" : "Explicit Euler");
        fprintf(out_file, "# Parameters: K=%.2f, r=%.4f, sigma=%.4f, T=%.2f, S_max=%.2f\n",
                params->K, params->r, params->sigma, params->T, params->S_max);
        fprintf(out_file, "# Fixed time steps: %d\n", 10000);
        fprintf(out_file, "# Columns: SpacePoints SpaceStep(dS) MaxError\n");
        fprintf(out_file, "# ----------------------------------------------\n");
    }
    
    // 固定时间步长，使用细网格确保时间误差可忽略
    int fixed_time_steps = 10000;
    int N_values[] = {50, 100, 200, 400, 800};
    int num_tests = sizeof(N_values) / sizeof(N_values[0]);
    
    double *errors = (double *)malloc(num_tests * sizeof(double));
    double *dS_values = (double *)malloc(num_tests * sizeof(double));
    
    // 保存原始参数
    int original_N = params->N;
    int original_time_steps = params->time_steps;
    char original_restart_file[256];
    strcpy(original_restart_file, params->restart_file);
    params->restart_file[0] = '\0';
    params->time_steps = fixed_time_steps;
    
    for (int i = 0; i < num_tests; i++) {
        params->N = N_values[i];
        
        double *u_final = NULL;
        int local_size, offset;
        int result = compute_option_price_verification(params, use_implicit, comm, 
                                                     &u_final, &local_size, &offset);
        
        if (result == 0) {
            double dS = params->S_max / (params->N - 1);
            double error = calculate_max_error(u_final, local_size, offset, params, dS, comm);
            
            if (rank == 0) {
                errors[i] = error;
                dS_values[i] = dS;
                
                // 写入输出文件
                fprintf(out_file, "%d %.6f %.6f\n", params->N, dS, error);
                
                // 控制台输出
                printf("空间网格数: %d, 误差: %.6f\n", params->N, error);
            }
            
            free(u_final);
        } else {
            if (rank == 0) {
                fprintf(out_file, "# Calculation diverged at N=%d\n", params->N);
                printf("计算在 N=%d 时发散\n", params->N);
            }
            break;
        }
    }
    
    // 计算并报告收敛阶
    if (rank == 0) {
        fprintf(out_file, "\n# Convergence Order Analysis\n");
        printf("\n收敛阶分析:\n");
        
        for (int i = 1; i < num_tests; i++) {
            if (errors[i] > 0 && errors[i-1] > 0) {
                double ratio = errors[i-1] / errors[i];
                double dS_ratio = dS_values[i-1] / dS_values[i];
                double convergence_order = log(ratio) / log(dS_ratio);
                
                char line[256];
                snprintf(line, sizeof(line), 
                         "# From N=%d to N=%d: Convergence order = %.4f (dS: %.4f -> %.4f, Error: %.4f -> %.4f)",
                         N_values[i-1], N_values[i], convergence_order,
                         dS_values[i-1], dS_values[i],
                         errors[i-1], errors[i]);
                
                fprintf(out_file, "%s\n", line);
                printf("%s\n", line);
            }
        }
        
        fprintf(out_file, "# ----------------------------------------------\n");
        fprintf(out_file, "# Test completed\n");
        
        // 添加总结信息
        fprintf(out_file, "\n# Test Summary\n");
        fprintf(out_file, "# Spatial discretization method: Finite Difference\n");
        fprintf(out_file, "# Theoretical expected convergence order: 2\n");
        fprintf(out_file, "# Actual average convergence order: %.4f\n", 
                (errors[1] > 0 && errors[0] > 0) ? 
                log(errors[0]/errors[1]) / log(dS_values[0]/dS_values[1]) : 0);
        fprintf(out_file, "# Verification result: %s\n", 
                (errors[num_tests-1] < 0.01) ? "PASS" : "FAIL");
        
        fclose(out_file);
        printf("空间收敛性测试结果已保存至: %s\n", "spatial_convergence_*.dat");
    }
    
    // 恢复原始参数
    params->N = original_N;
    params->time_steps = original_time_steps;
    strcpy(params->restart_file, original_restart_file);
    
    free(errors);
    free(dS_values);
}

// 时间收敛性测试 (输出为.dat文件)
void temporal_convergence_test(OptionParams *params, int use_implicit, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    // 创建输出文件
    FILE *out_file = NULL;
    if (rank == 0) {
        char out_filename[256];
        snprintf(out_filename, sizeof(out_filename), "temporal_convergence_%s.dat", 
                use_implicit ? "implicit" : "explicit");
        
        out_file = fopen(out_filename, "w");
        if (!out_file) {
            perror("Failed to create temporal convergence output file");
            return;
        }
        
        // 写入文件头
        fprintf(out_file, "# Temporal Convergence Test Results\n");
        fprintf(out_file, "# Method: %s\n", use_implicit ? "Implicit Euler" : "Explicit Euler");
        fprintf(out_file, "# Parameters: K=%.2f, r=%.4f, sigma=%.4f, T=%.2f, S_max=%.2f\n",
                params->K, params->r, params->sigma, params->T, params->S_max);
        fprintf(out_file, "# Fixed space points: %d\n", 1000);
        fprintf(out_file, "# Columns: TimeSteps TimeStep(dt) MaxError\n");
        fprintf(out_file, "# ----------------------------------------------\n");
    }
    
    // 固定空间网格数
    int fixed_N = 1000;
    int time_steps_values[] = {100, 200, 400, 800, 1600};
    int num_tests = sizeof(time_steps_values) / sizeof(time_steps_values[0]);
    
    double *errors = (double *)malloc(num_tests * sizeof(double));
    double *dt_values = (double *)malloc(num_tests * sizeof(double));
    
    // 保存原始参数
    int original_N = params->N;
    int original_time_steps = params->time_steps;
    char original_restart_file[256];
    strcpy(original_restart_file, params->restart_file);
    params->restart_file[0] = '\0';
    params->N = fixed_N;
    
    for (int i = 0; i < num_tests; i++) {
        params->time_steps = time_steps_values[i];
        
        double *u_final = NULL;
        int local_size, offset;
        int result = compute_option_price_verification(params, use_implicit, comm, 
                                                     &u_final, &local_size, &offset);
        
        if (result == 0) {
            double dS = params->S_max / (params->N - 1);
            double error = calculate_max_error(u_final, local_size, offset, params, dS, comm);
            
            if (rank == 0) {
                errors[i] = error;
                dt_values[i] = params->T / params->time_steps;
                
                // 写入输出文件
                fprintf(out_file, "%d %.6f %.6f\n", 
                       params->time_steps, dt_values[i], error);
                
                // 控制台输出
                printf("时间步数: %d, 误差: %.6f\n", params->time_steps, error);
            }
            
            free(u_final);
        } else {
            if (rank == 0) {
                fprintf(out_file, "# Calculation diverged at time steps=%d\n", params->time_steps);
                printf("计算在时间步数=%d 时发散\n", params->time_steps);
            }
            break;
        }
    }
    
    // 计算并报告收敛阶
    if (rank == 0) {
        fprintf(out_file, "\n# Convergence Order Analysis\n");
        printf("\n收敛阶分析:\n");
        
        for (int i = 1; i < num_tests; i++) {
            if (errors[i] > 0 && errors[i-1] > 0) {
                double ratio = errors[i-1] / errors[i];
                double dt_ratio = dt_values[i-1] / dt_values[i];
                double convergence_order = log(ratio) / log(dt_ratio);
                
                char line[256];
                snprintf(line, sizeof(line), 
                         "# From time steps=%d to %d: Convergence order = %.4f (dt: %.6f -> %.6f, Error: %.6f -> %.6f)",
                         time_steps_values[i-1], time_steps_values[i], convergence_order,
                         dt_values[i-1], dt_values[i],
                         errors[i-1], errors[i]);
                
                fprintf(out_file, "%s\n", line);
                printf("%s\n", line);
            }
        }
        
        fprintf(out_file, "# ----------------------------------------------\n");
        fprintf(out_file, "# Test completed\n");
        
        // 添加总结信息
        fprintf(out_file, "\n# Test Summary\n");
        fprintf(out_file, "# Time discretization method: %s\n", use_implicit ? "Implicit Euler" : "Explicit Euler");
        fprintf(out_file, "# Theoretical expected convergence order: %d\n", use_implicit ? 1 : 1);
        fprintf(out_file, "# Actual average convergence order: %.4f\n", 
                (errors[1] > 0 && errors[0] > 0) ? 
                log(errors[0]/errors[1]) / log(dt_values[0]/dt_values[1]) : 0);
        fprintf(out_file, "# Verification result: %s\n", 
                (errors[num_tests-1] < 0.01) ? "PASS" : "FAIL");
        
        fclose(out_file);
        printf("时间收敛性测试结果已保存至: %s\n", "temporal_convergence_*.dat");
    }
    
    // 恢复原始参数
    params->N = original_N;
    params->time_steps = original_time_steps;
    strcpy(params->restart_file, original_restart_file);
    
    free(errors);
    free(dt_values);
}

// 代码验证函数
void code_verification(OptionParams *params, int use_implicit, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    if (rank == 0) {
        printf("\n===== 代码验证 =====\n");
        printf("方法: %s\n", use_implicit ? "隐式欧拉" : "显式欧拉");
        printf("参数: K=%.2f, r=%.4f, sigma=%.4f, T=%.2f, S_max=%.2f\n",
              params->K, params->r, params->sigma, params->T, params->S_max);
    }
    
    // 进行空间收敛性测试
    spatial_convergence_test(params, use_implicit, comm);
    
    // 进行时间收敛性测试
    temporal_convergence_test(params, use_implicit, comm);
    
    if (rank == 0) {
        printf("代码验证完成\n");
    }
}

// ======================== 并行性能测试 ========================

// 固定问题规模并行性能测试
void fixed_size_scalability_test(OptionParams *params, int use_implicit, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    
    // 固定参数
    const int fixed_N = 10000;         // 固定全局网格点数
    const int fixed_time_steps = 100;   // 固定时间步数
    const int repeats = 3;              // 重复测试次数
    
    // 进程数列表
    const int proc_list[] = {1, 2, 4, 8, 16};
    const int num_tests = sizeof(proc_list)/sizeof(proc_list[0]);
    
    // 创建输出文件
    FILE *out_file = NULL;
    char out_filename[256];
    if (rank == 0) {
        snprintf(out_filename, sizeof(out_filename), 
                "fixed_size_scalability_%s.dat",
                use_implicit ? "implicit" : "explicit");
        out_file = fopen(out_filename, "w");
        if (!out_file) {
            perror("Failed to create fixed size scalability output file");
            return;
        }
        
        // 写入文件头
        fprintf(out_file, "# Fixed Problem Size Parallel Performance Test\n");
        fprintf(out_file, "# Method: %s\n", use_implicit ? "Implicit Euler" : "Explicit Euler");
        fprintf(out_file, "# Parameters: K=%.2f, r=%.4f, sigma=%.4f, T=%.2f, S_max=%.2f\n",
                params->K, params->r, params->sigma, params->T, params->S_max);
        fprintf(out_file, "# Fixed space points: %d\n", fixed_N);
        fprintf(out_file, "# Fixed time steps: %d\n", fixed_time_steps);
        fprintf(out_file, "# Test repeats: %d\n", repeats);
        fprintf(out_file, "# --------------------------------------------------------------\n");
        fprintf(out_file, "# Procs | SpacePoints | AssemblyTime(s) | SolverTime(s) | TotalTime(s) | Speedup | Efficiency(%%)\n");
        fprintf(out_file, "# --------------------------------------------------------------\n");
    }
    
    // 保存原始参数
    int original_N = params->N;
    int original_time_steps = params->time_steps;
    params->N = fixed_N;
    params->time_steps = fixed_time_steps;
    
    // 单进程基准时间
    double base_assembly_time = 0.0;
    double base_solver_time = 0.0;
    double base_total_time = 0.0;
    
    // 测试不同进程数
    for (int i = 0; i < num_tests; i++) {
        int num_procs = proc_list[i];
        
        // 创建子通信器
        MPI_Comm sub_comm;
        MPI_Comm_split(comm, rank < num_procs ? 0 : MPI_UNDEFINED, rank, &sub_comm);
        
        if (rank < num_procs) {
            double total_assembly = 0.0;
            double total_solver = 0.0;
            double total_total = 0.0;
            
            // 重复测试
            for (int r = 0; r < repeats; r++) {
                // 计时变量
                double assembly_start, assembly_end;
                double solver_start, solver_end;
                double total_start, total_end;
                
                // 重置参数
                params->current_time = 0.0;
                
                // 开始总计时
                MPI_Barrier(sub_comm);
                total_start = MPI_Wtime();
                
                // 模拟矩阵组装过程
                MPI_Barrier(sub_comm);
                assembly_start = MPI_Wtime();
                // 实际矩阵组装代码将放在这里
                assembly_end = MPI_Wtime();
                
                // 模拟求解过程
                MPI_Barrier(sub_comm);
                solver_start = MPI_Wtime();
                // 实际求解器代码将放在这里
                solver_end = MPI_Wtime();
                
                // 结束总计时
                MPI_Barrier(sub_comm);
                total_end = MPI_Wtime();
                
                // 累加时间
                total_assembly += (assembly_end - assembly_start);
                total_solver += (solver_end - solver_start);
                total_total += (total_end - total_start);
            }
            
            // 计算平均时间
            double avg_assembly = total_assembly / repeats;
            double avg_solver = total_solver / repeats;
            double avg_total = total_total / repeats;
            
            // 收集到主进程
            double assembly_time, solver_time, total_time;
            MPI_Reduce(&avg_assembly, &assembly_time, 1, MPI_DOUBLE, MPI_SUM, 0, sub_comm);
            MPI_Reduce(&avg_solver, &solver_time, 1, MPI_DOUBLE, MPI_SUM, 0, sub_comm);
            MPI_Reduce(&avg_total, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, sub_comm);
            
            // 主进程记录结果
            if (rank == 0) {
                // 计算加速比和效率
                double speedup = (i == 0) ? 1.0 : base_total_time / total_time;
                double efficiency = (speedup / num_procs) * 100;
                
                // 保存基准时间
                if (i == 0) {
                    base_assembly_time = assembly_time;
                    base_solver_time = solver_time;
                    base_total_time = total_time;
                }
                
                // 写入输出文件
                fprintf(out_file, "%d %d %.4f %.4f %.4f %.2f %.2f\n",
                       num_procs, fixed_N, assembly_time, solver_time, total_time,
                       speedup, efficiency);
                
                // 控制台输出
                printf("进程数: %d, 总时间: %.4fs, 加速比: %.2f, 效率: %.2f%%\n",
                       num_procs, total_time, speedup, efficiency);
            }
            
            MPI_Comm_free(&sub_comm);
        }
    }
    
    // 恢复原始参数
    params->N = original_N;
    params->time_steps = original_time_steps;
    
    if (rank == 0) {
        fprintf(out_file, "# --------------------------------------------------------------\n");
        fprintf(out_file, "# Test completed\n");
        fclose(out_file);
        printf("固定问题规模测试结果已保存至: %s\n", out_filename);
    }
}

// 固定每进程问题规模并行性能测试
void isogranular_scalability_test(OptionParams *params, int use_implicit, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    
    // 固定参数
    const int per_proc_N = 1000;        // 每进程网格点数
    const int fixed_time_steps = 100;    // 固定时间步数
    const int repeats = 3;               // 重复测试次数
    
    // 进程数列表
    const int proc_list[] = {1, 2, 4, 8, 16};
    const int num_tests = sizeof(proc_list)/sizeof(proc_list[0]);
    
    // 创建输出文件
    FILE *out_file = NULL;
    char out_filename[256];
    if (rank == 0) {
        snprintf(out_filename, sizeof(out_filename), 
                "isogranular_scalability_%s.dat",
                use_implicit ? "implicit" : "explicit");
        out_file = fopen(out_filename, "w");
        if (!out_file) {
            perror("Failed to create isogranular scalability output file");
            return;
        }
        
        // 写入文件头
        fprintf(out_file, "# Fixed Per-Processor Problem Size Parallel Performance Test\n");
        fprintf(out_file, "# Method: %s\n", use_implicit ? "Implicit Euler" : "Explicit Euler");
        fprintf(out_file, "# Parameters: K=%.2f, r=%.4f, sigma=%.4f, T=%.2f, S_max=%.2f\n",
                params->K, params->r, params->sigma, params->T, params->S_max);
        fprintf(out_file, "# Per-processor space points: %d\n", per_proc_N);
        fprintf(out_file, "# Fixed time steps: %d\n", fixed_time_steps);
        fprintf(out_file, "# Test repeats: %d\n", repeats);
        fprintf(out_file, "# --------------------------------------------------------------------------------\n");
        fprintf(out_file, "# Procs | GlobalSpace | PerProcSpace | AssemblyTime(s) | SolverTime(s) | TotalTime(s) | RelativeTime\n");
        fprintf(out_file, "# --------------------------------------------------------------------------------\n");
    }
    
    // 保存原始参数
    int original_time_steps = params->time_steps;
    params->time_steps = fixed_time_steps;
    
    // 单进程基准时间
    double base_total_time = 0.0;
    
    // 测试不同进程数
    for (int i = 0; i < num_tests; i++) {
        int num_procs = proc_list[i];
        int global_N = per_proc_N * num_procs;
        
        // 创建子通信器
        MPI_Comm sub_comm;
        MPI_Comm_split(comm, rank < num_procs ? 0 : MPI_UNDEFINED, rank, &sub_comm);
        
        if (rank < num_procs) {
            // 设置当前网格大小
            params->N = global_N;
            
            double total_assembly = 0.0;
            double total_solver = 0.0;
            double total_total = 0.0;
            
            // 重复测试
            for (int r = 0; r < repeats; r++) {
                // 计时变量
                double assembly_start, assembly_end;
                double solver_start, solver_end;
                double total_start, total_end;
                
                // 重置参数
                params->current_time = 0.0;
                
                // 开始总计时
                MPI_Barrier(sub_comm);
                total_start = MPI_Wtime();
                
                // 模拟矩阵组装过程
                MPI_Barrier(sub_comm);
                assembly_start = MPI_Wtime();
                // 实际矩阵组装代码将放在这里
                assembly_end = MPI_Wtime();
                
                // 模拟求解过程
                MPI_Barrier(sub_comm);
                solver_start = MPI_Wtime();
                // 实际求解器代码将放在这里
                solver_end = MPI_Wtime();
                
                // 结束总计时
                MPI_Barrier(sub_comm);
                total_end = MPI_Wtime();
                
                // 累加时间
                total_assembly += (assembly_end - assembly_start);
                total_solver += (solver_end - solver_start);
                total_total += (total_end - total_start);
            }
            
            // 计算平均时间
            double avg_assembly = total_assembly / repeats;
            double avg_solver = total_solver / repeats;
            double avg_total = total_total / repeats;
            
            // 收集到主进程
            double assembly_time, solver_time, total_time;
            MPI_Reduce(&avg_assembly, &assembly_time, 1, MPI_DOUBLE, MPI_SUM, 0, sub_comm);
            MPI_Reduce(&avg_solver, &solver_time, 1, MPI_DOUBLE, MPI_SUM, 0, sub_comm);
            MPI_Reduce(&avg_total, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, sub_comm);
            
            // 主进程记录结果
            if (rank == 0) {
                // 计算相对时间
                double relative_time = (i == 0) ? 1.0 : total_time / base_total_time;
                
                // 保存基准时间
                if (i == 0) {
                    base_total_time = total_time;
                }
                
                // 写入输出文件
                fprintf(out_file, "%d %d %d %.4f %.4f %.4f %.4f\n",
                       num_procs, global_N, per_proc_N, 
                       assembly_time, solver_time, total_time, relative_time);
                
                // 控制台输出
                printf("进程数: %d, 全局网格: %d, 总时间: %.4fs, 相对时间: %.4f\n",
                       num_procs, global_N, total_time, relative_time);
            }
            
            MPI_Comm_free(&sub_comm);
        }
    }
    
    // 恢复原始参数
    params->time_steps = original_time_steps;
    
    if (rank == 0) {
        fprintf(out_file, "# --------------------------------------------------------------------------------\n");
        fprintf(out_file, "# Test completed\n");
        fclose(out_file);
        printf("固定每进程问题规模测试结果已保存至: %s\n", out_filename);
    }
}

// ======================== 主函数 ========================

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // 默认参数
    OptionParams params = {
        .K = 100.0,
        .r = 0.05,
        .sigma = 0.3,
        .T = 1.0,
        .S_max = 500.0,
        .option_type = 0,
        .N = 1000,
        .time_steps = 1000,
        .save_interval = 100,
        .restart_file = "", 
        .current_time = 0.0,
        .stability_test = 0
    };
    
    int use_implicit = 1; // 默认使用隐式方法
    int stability_test = 0; // 是否进行稳定性测试
    int test_mode = 0;    // 测试模式
    int verification_mode = 0; // 代码验证模式
    int fixed_size_test = 0;
    int isogranular_test = 0;
    int post_mode = 0;    // 后处理模式
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-K") == 0) {
            params.K = atof(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0) {
            params.r = atof(argv[++i]);
        } else if (strcmp(argv[i], "-sigma") == 0) {
            params.sigma = atof(argv[++i]);
        } else if (strcmp(argv[i], "-T") == 0) {
            params.T = atof(argv[++i]);
        } else if (strcmp(argv[i], "-S_max") == 0) {
            params.S_max = atof(argv[++i]);
        } else if (strcmp(argv[i], "-option_type") == 0) {
            params.option_type = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-N") == 0) {
            params.N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-time_steps") == 0) {
            params.time_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-save_interval") == 0) {
            params.save_interval = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-restart_file") == 0) {
            strncpy(params.restart_file, argv[++i], sizeof(params.restart_file) - 1);
            params.restart_file[sizeof(params.restart_file) - 1] = '\0';
        } else if (strcmp(argv[i], "-explicit") == 0) {
            use_implicit = 0;
        } else if (strcmp(argv[i], "-test") == 0) {
            test_mode = 1;
        } else if (strcmp(argv[i], "-stability_test") == 0) {
            stability_test = 1;
            params.stability_test = 1;
        } else if (strcmp(argv[i], "-verification") == 0) {
            verification_mode = 1;
        } else if (strcmp(argv[i], "-fixed_size_test") == 0) {
            fixed_size_test = 1;
        } else if (strcmp(argv[i], "-isogranular_test") == 0) {
            isogranular_test = 1;
        } else if (strcmp(argv[i], "-post") == 0) {
            post_mode = 1;
        }
    }
    
    // 根据命令行参数选择功能
    if (stability_test) {
        // 进行数值稳定性测试
        numerical_stability_test(&params, use_implicit, MPI_COMM_WORLD);
    } else if (verification_mode) {
        // 进行代码验证
        code_verification(&params, use_implicit, MPI_COMM_WORLD);
    } else if (fixed_size_test) {
        // 固定问题规模并行性能测试
        fixed_size_scalability_test(&params, use_implicit, MPI_COMM_WORLD);
    } else if (isogranular_test) {
        // 固定每进程问题规模并行性能测试
        isogranular_scalability_test(&params, use_implicit, MPI_COMM_WORLD);
    } else if (post_mode) {
        
        // 运行后处理
        run_postprocessing(&params, use_implicit, MPI_COMM_WORLD);
    } else if (test_mode) {
        // TODO: 其他测试模式...
    } else {
        // 普通计算模式
        if (rank == 0) {
            printf("Running Black-Scholes solver with parameters:\n");
            printf("  Option type:       %s\n", params.option_type ? "Put" : "Call");
            printf("  Strike price (K):  %.2f\n", params.K);
            printf("  Risk-free rate (r): %.4f\n", params.r);
            printf("  Volatility (sigma): %.4f\n", params.sigma);
            printf("  Maturity (T):      %.2f\n", params.T);
            printf("  Asset max (S_max): %.2f\n", params.S_max);
            printf("  Space points (N):  %d\n", params.N);
            printf("  Time steps:        %d\n", params.time_steps);
            printf("  Method:            %s\n", use_implicit ? "Implicit Euler" : "Explicit Euler");
            printf("  Save interval:     %d\n", params.save_interval);
            if (strlen(params.restart_file)) {
                printf("  Restart file:      %s\n", params.restart_file);
            } else {
                printf("  Restart file:      None\n");
            }
        }
        
        int result = compute_option_price(&params, use_implicit, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (result == 0) {
                printf("Calculation completed successfully.\n");
            } else {
                printf("Calculation diverged!\n");
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
