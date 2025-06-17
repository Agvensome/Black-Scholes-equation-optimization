#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <errno.h>
#include <time.h>

// ���ڲ����ṹ��
typedef struct {
    double K;           // ��Ȩ��
    double r;           // �޷�������
    double sigma;       // ������
    double T;           // ����ʱ��
    double S_max;       // �ʲ��۸�����
    int option_type;    // 0=����, 1=����
    int N;              // �ռ�������
    int time_steps;     // ʱ�䲽��
    int save_interval;  // ������
    char restart_file[256]; // �����ļ���
    double current_time; // ��ǰʱ��
    int stability_test; // �Ƿ�����ȶ��Բ���
} OptionParams;

// �����״̬
typedef struct {
    double *u;          // ��ǰ������
    double *u_old;      // ��һʱ�䲽������
    int current_step;    // ��ǰ����
} SolverState;

// ������߽�ֵ
double left_boundary_value(double t, OptionParams *params) {
    if (params->option_type == 1) { // ������Ȩ
        return params->K * exp(-params->r * (params->T - t));
    }
    return 0.0; // ������Ȩ
}

// �����ұ߽�ֵ
double right_boundary_value(double t, OptionParams *params) {
    if (params->option_type == 0) { // ������Ȩ
        return params->S_max - params->K * exp(-params->r * (params->T - t));
    }
    return 0.0; // ������Ȩ
}

// ���ó�ʼ����
void set_initial_condition(double *u, int local_size, int offset, OptionParams *params) {
    double dS = params->S_max / (params->N - 1);
    for (int i = 0; i < local_size; i++) {
        double S = (offset + i) * dS;
        if (params->option_type == 0) { // ������Ȩ
            u[i] = (S > params->K) ? S - params->K : 0.0;
        } else { // ������Ȩ
            u[i] = (S < params->K) ? params->K - S : 0.0;
        }
    }
}

// Ӧ�ñ߽�����
void apply_boundary_condition(double *u, int local_size, int rank, int size, 
                             double t, OptionParams *params) {
    // ��߽� (rank 0�ĵ�һ����)
    if (rank == 0) {
        u[0] = left_boundary_value(t, params);
    }
    
    // �ұ߽� (���һ��rank�����һ����)
    if (rank == size - 1) {
        u[local_size - 1] = right_boundary_value(t, params);
    }
}

// �����߽�����
void exchange_boundary_data(double *u, int local_size, int rank, int size, MPI_Comm comm) {
    double left_bound = 0.0, right_bound = 0.0;
    MPI_Status status;
    
    // �����ұ߽����һ�����̣�������߽����һ������
    if (rank < size - 1) {
        MPI_Send(&u[local_size - 1], 1, MPI_DOUBLE, rank + 1, 0, comm);
    }
    if (rank > 0) {
        MPI_Recv(&left_bound, 1, MPI_DOUBLE, rank - 1, 0, comm, &status);
        u[0] = left_bound;
    }
    
    // ������߽����һ�����̣������ұ߽����һ������
    if (rank > 0) {
        MPI_Send(&u[0], 1, MPI_DOUBLE, rank - 1, 1, comm);
    }
    if (rank < size - 1) {
        MPI_Recv(&right_bound, 1, MPI_DOUBLE, rank + 1, 1, comm, &status);
        u[local_size - 1] = right_bound;
    }
}

// ��ʽŷ��ʱ�䲽��
void explicit_euler_step(double *u, double *u_old, int local_size, double dS, double dt, 
                         OptionParams *params, int rank, int size, MPI_Comm comm) {
    double alpha, beta, gamma;
    double S;
    
    // ���ƾ�ֵ
    memcpy(u_old, u, local_size * sizeof(double));
    
    // �����ڲ���
    for (int i = 1; i < local_size - 1; i++) {
        S = (rank * (params->N / size) + i) * dS;
        alpha = 0.5 * params->sigma * params->sigma * S * S;
        beta = params->r * S;
        gamma = params->r;
        
        double d2V_dS2 = (u_old[i+1] - 2*u_old[i] + u_old[i-1]) / (dS * dS);
        double dV_dS = (u_old[i+1] - u_old[i-1]) / (2 * dS);
        
        u[i] = u_old[i] + dt * (alpha * d2V_dS2 + beta * dV_dS - gamma * u_old[i]);
    }
    
    // �����߽�����
    exchange_boundary_data(u, local_size, rank, size, comm);
    
    // Ӧ�ñ߽�����
    apply_boundary_condition(u, local_size, rank, size, params->current_time + dt, params);
}

// ���ԽǾ������ (Thomas�㷨)
void solve_tridiagonal(double *a, double *b, double *c, double *d, double *x, int n) {
    // ǰ����Ԫ
    for (int i = 1; i < n; i++) {
        double m = a[i] / b[i-1];
        b[i] = b[i] - m * c[i-1];
        d[i] = d[i] - m * d[i-1];
    }
    
    // �ش�
    x[n-1] = d[n-1] / b[n-1];
    for (int i = n - 2; i >= 0; i--) {
        x[i] = (d[i] - c[i] * x[i+1]) / b[i];
    }
}

// ��ʽŷ��ʱ�䲽��
void implicit_euler_step(double *u, double *u_old, int local_size, double dS, double dt, 
                         OptionParams *params, int rank, int size, MPI_Comm comm) {
    double alpha, beta, gamma;
    double S;
    
    // ���ƾ�ֵ
    memcpy(u_old, u, local_size * sizeof(double));
    
    // �������Խ�ϵͳ�ڴ�
    double *a = (double *)malloc(local_size * sizeof(double)); // �¶Խ���
    double *b = (double *)malloc(local_size * sizeof(double)); // ���Խ���
    double *c = (double *)malloc(local_size * sizeof(double)); // �϶Խ���
    double *d = (double *)malloc(local_size * sizeof(double)); // �Ҳ�����
    
    // ��������ϵͳ
    for (int i = 0; i < local_size; i++) {
        S = (rank * (params->N / size) + i) * dS;
        alpha = 0.5 * params->sigma * params->sigma * S * S;
        beta = params->r * S;
        gamma = params->r;
        
        if (i == 0) {
            // ��߽�
            a[i] = 0.0;
            b[i] = 1.0;
            c[i] = 0.0;
            d[i] = left_boundary_value(params->current_time + dt, params);
        } else if (i == local_size - 1) {
            // �ұ߽�
            a[i] = 0.0;
            b[i] = 1.0;
            c[i] = 0.0;
            d[i] = right_boundary_value(params->current_time + dt, params);
        } else {
            // �ڲ���
            double coeff1 = alpha / (dS * dS) - beta / (2 * dS);
            double coeff2 = 1.0/dt + 2.0 * alpha / (dS * dS) + gamma;
            double coeff3 = alpha / (dS * dS) + beta / (2 * dS);
            
            a[i] = -dt * coeff1;
            b[i] = 1.0 + dt * coeff2;
            c[i] = -dt * coeff3;
            d[i] = u_old[i];
        }
    }
    
    // ������Խ�ϵͳ
    solve_tridiagonal(a, b, c, d, u, local_size);
    
    // �ͷ��ڴ�
    free(a);
    free(b);
    free(c);
    free(d);
    
    // �����߽�����
    exchange_boundary_data(u, local_size, rank, size, comm);
    
    // Ӧ�ñ߽�����
    apply_boundary_condition(u, local_size, rank, size, params->current_time + dt, params);
}

// ���������ļ�
void save_restart_file(SolverState *state, int local_size, int offset, 
                       OptionParams *params, int rank) {
    // ���û��ָ�������ļ������򲻱���
    if (strlen(params->restart_file) == 0) return;
    
    char filename[256];
    snprintf(filename, sizeof(filename), "%s_%d.dat", params->restart_file, rank);
    
    FILE *fp = fopen(filename, "wb");
    if (fp) {
        // ����Ԫ����
        fwrite(&state->current_step, sizeof(int), 1, fp);
        fwrite(&params->current_time, sizeof(double), 1, fp);
        fwrite(&offset, sizeof(int), 1, fp);
        fwrite(&local_size, sizeof(int), 1, fp);
        
        // ���������
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

// ���������ļ�
int load_restart_file(SolverState *state, int *local_size, int *offset, 
                      OptionParams *params, int rank) {
    // ���û��ָ�������ļ������򲻼���
    if (strlen(params->restart_file) == 0) return 0;
    
    char filename[256];
    snprintf(filename, sizeof(filename), "%s_%d.dat", params->restart_file, rank);
    
    FILE *fp = fopen(filename, "rb");
    if (fp) {
        // ��ȡԪ����
        fread(&state->current_step, sizeof(int), 1, fp);
        fread(&params->current_time, sizeof(double), 1, fp);
        fread(offset, sizeof(int), 1, fp);
        fread(local_size, sizeof(int), 1, fp);
        
        // �����ڴ�
        state->u = (double *)malloc(*local_size * sizeof(double));
        state->u_old = (double *)malloc(*local_size * sizeof(double));
        
        // ��ȡ������
        fread(state->u, sizeof(double), *local_size, fp);
        memcpy(state->u_old, state->u, *local_size * sizeof(double));
        
        fclose(fp);
        if (rank == 0 && !params->stability_test) {
            printf("Loaded restart file: %s\n", filename);
        }
        return 1; // �ɹ�����
    } else {
        if (rank == 0 && !params->stability_test) {
            printf("Restart file not found: %s. Starting from initial conditions.\n", filename);
        }
        return 0; // �ļ�δ�ҵ�
    }
}

// �����ֵ���Ƿ�ɢ
int check_divergence(double *u, int local_size, MPI_Comm comm) {
    int local_diverged = 0;
    int global_diverged = 0;
    
    // ��鱾���������Ƿ�ɢ
    for (int i = 0; i < local_size; i++) {
        if (isnan(u[i]) || isinf(u[i]) || fabs(u[i]) > 1e100) {
            local_diverged = 1;
            break;
        }
    }
    
    // ȫ�ּ���Ƿ�ɢ
    MPI_Allreduce(&local_diverged, &global_diverged, 1, MPI_INT, MPI_LOR, comm);
    
    return global_diverged;
}

// ������Ȩ�۸�
int compute_option_price(OptionParams *params, int use_implicit, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // ����ÿ�����̵ľֲ������С
    int local_size = params->N / size;
    int remainder = params->N % size;
    
    // ���������������
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
    
    // ��ʼ�������״̬
    SolverState state;
    state.current_step = 0;
    params->current_time = 0.0; // ��ʼ����ǰʱ��
    
    // ����Ƿ���Ҫ����
    int restart_loaded = 0;
    if (strlen(params->restart_file)) {
        restart_loaded = load_restart_file(&state, &local_size, &offset, params, rank);
    }
    
    if (!restart_loaded) {
        // �����ڴ�
        state.u = (double *)malloc(local_size * sizeof(double));
        state.u_old = (double *)malloc(local_size * sizeof(double));
        
        // ���ó�ʼ����
        set_initial_condition(state.u, local_size, offset, params);
        memcpy(state.u_old, state.u, local_size * sizeof(double));
        
        // Ӧ�ó�ʼ�߽�����
        apply_boundary_condition(state.u, local_size, rank, size, 0.0, params);
    }
    
    // ʱ�䲽��ѭ��
    int diverged = 0;
    for (int step = 0; step < params->time_steps; step++) {
        if (diverged) break;
        
        if (use_implicit) {
            implicit_euler_step(state.u, state.u_old, local_size, dS, dt, params, rank, size, comm);
        } else {
            explicit_euler_step(state.u, state.u_old, local_size, dS, dt, params, rank, size, comm);
        }
        
        // ����Ƿ�ɢ
        if (step % 10 == 0) {
            diverged = check_divergence(state.u, local_size, comm);
            if (diverged) {
                if (rank == 0) {
                    printf("������ʱ�䲽 %d (ʱ�� %.4f) ��ɢ\n", state.current_step, params->current_time);
                }
                break;
            }
        }
        
        state.current_step++;
        params->current_time += dt;
        
        // ���ڱ��������ļ�
        if (params->save_interval > 0 && state.current_step % params->save_interval == 0) {
            save_restart_file(&state, local_size, offset, params, rank);
        }
    }
    
    // �������ս��
    if (params->save_interval > 0 && strlen(params->restart_file) && !diverged) {
        save_restart_file(&state, local_size, offset, params, rank);
    }
    
    // �����ڴ�
    free(state.u);
    free(state.u_old);
    
    return diverged ? 1 : 0;
}

// ��ֵ�ȶ��Բ��Ժ���
// ��ֵ�ȶ��Բ��Ժ���
void numerical_stability_test(OptionParams *params, int use_implicit, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    // ���ٿռ�������������� ��S
    const int fixed_N = 100;
    params->N = fixed_N;
    
    double dS = params->S_max / (fixed_N - 1);
    double stability_bound = (dS * dS) / (params->sigma * params->sigma * params->S_max * params->S_max);
    
    // �����̴�������ļ�
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
        
        printf("��ֵ�ȶ��Բ��� - %s����\n", use_implicit ? "��ʽ" : "��ʽ");
        printf("�ռ�������� N = %d\n", fixed_N);
        printf("�ռ䲽�� ��S = %.6f\n", dS);
        printf("�����ȶ��Ա߽� ��t < %.6f\n", stability_bound);
        printf("ʱ�䲽�� | ʱ�䲽�� ��t | ��t/�߽� | �ȶ���\n");
        printf("-------------------------------------------\n");
    }
    
    // ���Բ�������
    const int num_tests = 15;
    double time_step_ratios[num_tests];
    double min_ratio = 0.1;
    double max_ratio = use_implicit ? 10.0 : 3.0;
    double step = (max_ratio - min_ratio) / (num_tests - 1);
    
    for (int i = 0; i < num_tests; i++) {
        time_step_ratios[i] = min_ratio + i * step;
    }
    
    // ִ�в���
    for (int i = 0; i < num_tests; i++) {
        double ratio = time_step_ratios[i];
        double dt = ratio * stability_bound;
        params->time_steps = (int)ceil(params->T / dt);
        if (params->time_steps < 1) params->time_steps = 1;
        
        int result = compute_option_price(params, use_implicit, comm);
        int diverged;
        MPI_Reduce(&result, &diverged, 1, MPI_INT, MPI_MAX, 0, comm);
        
        // ������������
        if (rank == 0) {
            printf("%6d | %10.6f | %7.2f | %s\n", 
                   params->time_steps, dt, ratio, 
                   diverged ? "��ɢ" : "�ȶ�");
            
            // д���ļ�
            if (fp) {
                fprintf(fp, "%d %.6f %.6f %d\n", 
                        params->time_steps, dt, ratio, diverged);
                fflush(fp);  // ȷ��ÿ��д���ˢ��
            }
            
            fflush(stdout);
        }
    }
    
    // ������Դ
    if (rank == 0) {
        printf("-------------------------------------------\n");
        printf("������ɡ�����ѱ�����: stability_test_%s.dat\n", 
              use_implicit ? "implicit" : "explicit");
        
        if (fp) {
            fclose(fp);
        }
    }
}

// ������
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Ĭ�ϲ���
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
    
    int use_implicit = 1; // Ĭ��ʹ����ʽ����
    int stability_test = 0; // �Ƿ�����ȶ��Բ���
    int test_mode = 0;    // ����ģʽ
    
    // ���������в���
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
        // ������ֵ�ȶ��Բ���
        numerical_stability_test(&params, use_implicit, MPI_COMM_WORLD);
    } else if (test_mode) {
        // ��������ģʽ...
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
