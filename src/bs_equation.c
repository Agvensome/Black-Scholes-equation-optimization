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

