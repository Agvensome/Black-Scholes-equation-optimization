#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include <errno.h>

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
