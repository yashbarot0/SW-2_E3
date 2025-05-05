#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MATRIX_DIM 20
#define BLOCK_SIZE 5
#define NUM_PROCESSES 4

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != NUM_PROCESSES) {
        if (rank == 0) fprintf(stderr, "Run with 4 processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read matrix
    MPI_File mat_file;
    MPI_File_open(MPI_COMM_WORLD, "mat-d20-b5-p4.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &mat_file);
    int dim;
    MPI_File_read_at(mat_file, 0, &dim, 1, MPI_INT, MPI_STATUS_IGNORE);
    if (dim != MATRIX_DIM) {
        if (rank == 0) fprintf(stderr, "Matrix dimension error.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double *A_local = malloc(MATRIX_DIM * BLOCK_SIZE * sizeof(double));
    MPI_Offset mat_offset = 4 + rank * MATRIX_DIM * BLOCK_SIZE * sizeof(double);
    MPI_File_read_at(mat_file, mat_offset, A_local, MATRIX_DIM * BLOCK_SIZE, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&mat_file);

    // Read vector
    MPI_File x_file;
    MPI_File_open(MPI_COMM_WORLD, "x-d20.txt.bin", MPI_MODE_RDONLY, MPI_INFO_NULL, &x_file);
    int x_len;
    MPI_File_read_at(x_file, 0, &x_len, 1, MPI_INT, MPI_STATUS_IGNORE);
    if (x_len != MATRIX_DIM) {
        if (rank == 0) fprintf(stderr, "Vector length error.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double *x_local = malloc(BLOCK_SIZE * sizeof(double));
    MPI_Offset x_offset = 4 + rank * BLOCK_SIZE * sizeof(double);
    MPI_File_read_at(x_file, x_offset, x_local, BLOCK_SIZE, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&x_file);

    // Compute local Ax part
    double *local_result = calloc(MATRIX_DIM, sizeof(double));
    for (int i = 0; i < MATRIX_DIM; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            local_result[i] += A_local[i * BLOCK_SIZE + j] * x_local[j];
        }
    }

    // Gather results at root
    double *global_result = NULL;
    if (rank == 0) global_result = malloc(MATRIX_DIM * sizeof(double));
    MPI_Reduce(local_result, global_result, MATRIX_DIM, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Output
    if (rank == 0) {
        printf("Ax = [");
        for (int i = 0; i < MATRIX_DIM; i++) printf("%.1f ", global_result[i]);
        printf("]\n");
        free(global_result);
    }

    free(A_local);
    free(x_local);
    free(local_result);
    MPI_Finalize();
    return 0;
}