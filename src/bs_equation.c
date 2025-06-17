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

// 数值稳定性测试函数
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

// 主函数
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
        }
    }
    
    if (stability_test) {
        // 进行数值稳定性测试
        numerical_stability_test(&params, use_implicit, MPI_COMM_WORLD);
    } else if (test_mode) {
        // 其他测试模式...
    } else {
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
