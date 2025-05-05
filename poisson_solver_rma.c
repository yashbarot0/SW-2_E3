/**
 * Poisson Solver with 2D Processor Decomposition
 * 
 * This program solves the 2D Poisson equation using a 5-point stencil finite difference method.
 * It implements three versions of ghost/halo exchange:
 * 1. Traditional MPI message passing (Send/Recv)
 * 2. RMA with MPI_Win_fence synchronization
 * 3. RMA with general active target synchronization
 * 
 * Compile with: mpicc -o poisson_solver_rma poisson_solver_rma.c -lm
 * Run with: mpiexec -n <num_procs> ./poisson_solver_rma <grid_size> <max_iterations> <exchange_method>
 * 
 * exchange_method: 0 = Message Passing, 1 = RMA with fence, 2 = RMA with general active target
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <string.h>
 #include <mpi.h>
 
 #define MAX_ITERATIONS 10000
 #define TOLERANCE 1e-6
 
 // Global variables for MPI ranks and dimensions
 int rank, size;
 int dims[2] = {0, 0};
 int coords[2];
 int north, south, east, west;
 MPI_Comm cart_comm;
 
 // Function prototypes
 void allocate_arrays(double ***u, double ***u_new, double ***f, int local_rows, int local_cols);
 void deallocate_arrays(double **u, double **u_new, double **f);
 void initialize_grid(double **u, double **u_new, double **f, int local_rows, int local_cols, 
                     int row_start, int col_start, int global_rows, int global_cols);
 double compute_residual(double **u, double **f, int local_rows, int local_cols, double h);
 void jacobi_iteration(double **u, double **u_new, double **f, int local_rows, int local_cols, double h);
 void exchange_halos_mpi(double **u, int local_rows, int local_cols, MPI_Datatype col_type);
 void exchange_halos_rma_fence(double **u, int local_rows, int local_cols, MPI_Win win, double *win_mem);
 void exchange_halos_rma_gats(double **u, int local_rows, int local_cols, MPI_Win win, double *win_mem,
                             int *neighbors, int num_neighbors);
 void copy_array(double **dst, double **src, int local_rows, int local_cols);
 
 int main(int argc, char *argv[]) {
     int global_rows, global_cols;
     int local_rows, local_cols;
     int row_start, col_start;
     double **u = NULL, **u_new = NULL, **f = NULL;
     double h, global_residual, local_residual;
     int iterations = 0;
     int max_iterations = MAX_ITERATIONS;
     int exchange_method = 0;  // 0: MPI, 1: RMA fence, 2: RMA GATS
     double start_time, setup_time, solve_time, end_time;
     
     MPI_Init(&argc, &argv);
     start_time = MPI_Wtime();
     
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
     
     // Parse command line arguments
     if (argc < 2) {
         if (rank == 0) {
             printf("Usage: %s <grid_size> [max_iterations] [exchange_method]\n", argv[0]);
             printf("exchange_method: 0 = Message Passing, 1 = RMA with fence, 2 = RMA with general active target\n");
         }
         MPI_Finalize();
         return 1;
     }
     
     global_rows = global_cols = atoi(argv[1]);
     if (argc >= 3) max_iterations = atoi(argv[2]);
     if (argc >= 4) exchange_method = atoi(argv[3]);
     
     // Create 2D Cartesian communicator
     MPI_Dims_create(size, 2, dims);
     int periods[2] = {0, 0};
     MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
     MPI_Cart_coords(cart_comm, rank, 2, coords);
     
     // Get neighboring ranks
     MPI_Cart_shift(cart_comm, 0, 1, &north, &south);  // y-direction
     MPI_Cart_shift(cart_comm, 1, 1, &west, &east);    // x-direction
     
     // Calculate local grid dimensions
     local_rows = global_rows / dims[0];
     local_cols = global_cols / dims[1];
     if (coords[0] < global_rows % dims[0]) local_rows++;
     if (coords[1] < global_cols % dims[1]) local_cols++;
     
     // Calculate starting indices in the global grid
     row_start = coords[0] * (global_rows / dims[0]);
     if (coords[0] < global_rows % dims[0]) row_start += coords[0];
     else row_start += global_rows % dims[0];
     
     col_start = coords[1] * (global_cols / dims[1]);
     if (coords[1] < global_cols % dims[1]) col_start += coords[1];
     else col_start += global_cols % dims[1];
     
     h = 1.0 / (global_rows + 1);  // Grid spacing
     
     // Allocate arrays with ghost points
     allocate_arrays(&u, &u_new, &f, local_rows + 2, local_cols + 2);
     
     // Initialize grid
     initialize_grid(u, u_new, f, local_rows, local_cols, row_start, col_start, global_rows, global_cols);
     
     // Create MPI datatype for sending/receiving columns
     MPI_Datatype col_type;
     MPI_Type_vector(local_rows, 1, local_cols + 2, MPI_DOUBLE, &col_type);
     MPI_Type_commit(&col_type);
     
     // Setup for RMA operations
     MPI_Win win;
     double *win_mem = NULL;
     int *neighbors = NULL;
     int num_neighbors = 0;
     
     if (exchange_method > 0) {
         // Allocate window memory for RMA operations (includes ghost cells)
         size_t local_size = (local_rows + 2) * (local_cols + 2) * sizeof(double);
         win_mem = (double *)malloc(local_size);
         if (win_mem == NULL) {
             fprintf(stderr, "Error: Failed to allocate window memory\n");
             MPI_Abort(MPI_COMM_WORLD, 1);
         }
         
         MPI_Win_create(win_mem, local_size, sizeof(double), 
                       MPI_INFO_NULL, cart_comm, &win);
         
         // For general active target sync, collect list of neighbors
         if (exchange_method == 2) {
             neighbors = (int *)malloc(4 * sizeof(int));
             if (neighbors == NULL) {
                 fprintf(stderr, "Error: Failed to allocate neighbors array\n");
                 MPI_Abort(MPI_COMM_WORLD, 1);
             }
             
             if (north != MPI_PROC_NULL) neighbors[num_neighbors++] = north;
             if (south != MPI_PROC_NULL) neighbors[num_neighbors++] = south;
             if (east != MPI_PROC_NULL) neighbors[num_neighbors++] = east;
             if (west != MPI_PROC_NULL) neighbors[num_neighbors++] = west;
         }
     }
     
     setup_time = MPI_Wtime() - start_time;
     solve_time = MPI_Wtime();
     
     // Main Jacobi iteration loop
     while (iterations < max_iterations) {
         // Perform Jacobi iteration
         jacobi_iteration(u, u_new, f, local_rows, local_cols, h);
         
         // Exchange halos based on selected method
         if (exchange_method == 0) {
             exchange_halos_mpi(u_new, local_rows, local_cols, col_type);
         } else if (exchange_method == 1) {
                 exchange_halos_rma_fence(u_new, local_rows, local_cols, win, win_mem);
         } else {
             exchange_halos_rma_gats(u_new, local_rows, local_cols, win, win_mem, neighbors, num_neighbors);
         }
         
         // Calculate residual
         local_residual = compute_residual(u_new, f, local_rows, local_cols, h);
         MPI_Allreduce(&local_residual, &global_residual, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
         global_residual = sqrt(global_residual);
         
         // Check convergence
         if (global_residual < TOLERANCE) {
             break;
         }
         
         // Swap pointers for next iteration
         double **temp = u;
         u = u_new;
         u_new = temp;
         
         iterations++;
         
         // Print progress every 100 iterations
         if (rank == 0 && iterations % 100 == 0) {
             printf("Iteration %d, Residual: %e\n", iterations, global_residual);
         }
     }
     
     solve_time = MPI_Wtime() - solve_time;
     end_time = MPI_Wtime() - start_time;
     
     // Print results
     if (rank == 0) {
         printf("\nSolution converged after %d iterations with residual %e\n", 
                iterations, global_residual);
         printf("Grid size: %d x %d, Processor grid: %d x %d\n", 
                global_rows, global_cols, dims[0], dims[1]);
         printf("Exchange method: %d (%s)\n", 
                exchange_method, 
                exchange_method == 0 ? "MPI Send/Recv" : 
                exchange_method == 1 ? "RMA with fence" : "RMA with general active target");
         printf("Setup time: %.4f s\n", setup_time);
         printf("Solve time: %.4f s\n", solve_time);
         printf("Total time: %.4f s\n", end_time);
     }
     
     // Clean up
     MPI_Type_free(&col_type);
     if (exchange_method > 0) {
         MPI_Win_free(&win);
         free(win_mem);
         if (exchange_method == 2) {
             free(neighbors);
         }
     }
     deallocate_arrays(u, u_new, f);
     MPI_Comm_free(&cart_comm);
     MPI_Finalize();
     
     return 0;
 }
 
 /**
  * Allocate 2D arrays with ghost cells
  */
 void allocate_arrays(double ***u, double ***u_new, double ***f, int local_rows, int local_cols) {
     *u = (double **)malloc(local_rows * sizeof(double *));
     *u_new = (double **)malloc(local_rows * sizeof(double *));
     *f = (double **)malloc(local_rows * sizeof(double *));
     
     if (*u == NULL || *u_new == NULL || *f == NULL) {
         fprintf(stderr, "Error: Failed to allocate array pointers\n");
         MPI_Abort(MPI_COMM_WORLD, 1);
     }
     
     double *u_data = (double *)calloc(local_rows * local_cols, sizeof(double));
     double *u_new_data = (double *)calloc(local_rows * local_cols, sizeof(double));
     double *f_data = (double *)calloc(local_rows * local_cols, sizeof(double));
     
     if (u_data == NULL || u_new_data == NULL || f_data == NULL) {
         fprintf(stderr, "Error: Failed to allocate array data\n");
         MPI_Abort(MPI_COMM_WORLD, 1);
     }
     
     for (int i = 0; i < local_rows; i++) {
         (*u)[i] = &u_data[i * local_cols];
         (*u_new)[i] = &u_new_data[i * local_cols];
         (*f)[i] = &f_data[i * local_cols];
     }
 }
 
 /**
  * Deallocate arrays
  */
 void deallocate_arrays(double **u, double **u_new, double **f) {
     free(u[0]);
     free(u_new[0]);
     free(f[0]);
     free(u);
     free(u_new);
     free(f);
 }
 
 /**
  * Initialize grid values
  */
 void initialize_grid(double **u, double **u_new, double **f, int local_rows, int local_cols, 
                     int row_start, int col_start, int global_rows, int global_cols) {
     // Initialize all arrays to zero
     memset(u[0], 0, (local_rows + 2) * (local_cols + 2) * sizeof(double));
     memset(u_new[0], 0, (local_rows + 2) * (local_cols + 2) * sizeof(double));
     memset(f[0], 0, (local_rows + 2) * (local_cols + 2) * sizeof(double));
     
     // Set source term for f (right-hand side of Poisson equation)
     double h = 1.0 / (global_rows + 1);
     double x_center = 0.5;
     double y_center = 0.5;
     double sigma = 0.1;
     
     for (int i = 1; i <= local_rows; i++) {
         for (int j = 1; j <= local_cols; j++) {
             double x = (col_start + j) * h;
             double y = (row_start + i) * h;
             
             // Gaussian source
             f[i][j] = exp(-(pow(x - x_center, 2) + pow(y - y_center, 2)) / (2 * sigma * sigma));
         }
     }
 }
 
 /**
  * Compute residual of current solution
  */
 double compute_residual(double **u, double **f, int local_rows, int local_cols, double h) {
     double residual = 0.0;
     double h_squared = h * h;
     
     for (int i = 1; i <= local_rows; i++) {
         for (int j = 1; j <= local_cols; j++) {
             double laplacian = (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - 4.0 * u[i][j]) / h_squared;
             double res = laplacian - f[i][j];
             residual += res * res;
         }
     }
     
     return residual;
 }
 
 /**
  * Perform one Jacobi iteration
  */
 void jacobi_iteration(double **u, double **u_new, double **f, int local_rows, int local_cols, double h) {
     double h_squared = h * h;
     
     for (int i = 1; i <= local_rows; i++) {
         for (int j = 1; j <= local_cols; j++) {
             u_new[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - h_squared * f[i][j]);
         }
     }
 }
 
 /**
  * Exchange halos using MPI message passing
  */
 void exchange_halos_mpi(double **u, int local_rows, int local_cols, MPI_Datatype col_type) {
     MPI_Request requests[8];
     MPI_Status statuses[8];
     int req_count = 0;
     
     // Send to north, receive from south
     if (north != MPI_PROC_NULL) {
         MPI_Isend(&u[1][1], local_cols, MPI_DOUBLE, north, 0, cart_comm, &requests[req_count++]);
     }
     if (south != MPI_PROC_NULL) {
         MPI_Irecv(&u[local_rows+1][1], local_cols, MPI_DOUBLE, south, 0, cart_comm, &requests[req_count++]);
     }
     
     // Send to south, receive from north
     if (south != MPI_PROC_NULL) {
         MPI_Isend(&u[local_rows][1], local_cols, MPI_DOUBLE, south, 1, cart_comm, &requests[req_count++]);
     }
     if (north != MPI_PROC_NULL) {
         MPI_Irecv(&u[0][1], local_cols, MPI_DOUBLE, north, 1, cart_comm, &requests[req_count++]);
     }
     
     // Send to west, receive from east
     if (west != MPI_PROC_NULL) {
         MPI_Isend(&u[1][1], 1, col_type, west, 2, cart_comm, &requests[req_count++]);
     }
     if (east != MPI_PROC_NULL) {
         MPI_Irecv(&u[1][local_cols+1], 1, col_type, east, 2, cart_comm, &requests[req_count++]);
     }
     
     // Send to east, receive from west
     if (east != MPI_PROC_NULL) {
         MPI_Isend(&u[1][local_cols], 1, col_type, east, 3, cart_comm, &requests[req_count++]);
     }
     if (west != MPI_PROC_NULL) {
         MPI_Irecv(&u[1][0], 1, col_type, west, 3, cart_comm, &requests[req_count++]);
     }
     
     MPI_Waitall(req_count, requests, statuses);
 }
 
 /**
  * Exchange halos using RMA with MPI_Win_fence synchronization
  */
 void exchange_halos_rma_fence(double **u, int local_rows, int local_cols, MPI_Win win, double *win_mem) {
     int local_size = (local_rows + 2) * (local_cols + 2);
     
     // Copy local array to window memory
     for (int i = 0; i <= local_rows + 1; i++) {
         for (int j = 0; j <= local_cols + 1; j++) {
             win_mem[i * (local_cols + 2) + j] = u[i][j];
         }
     }
     
     // Ensure all local operations are done before starting RMA
     MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
     
     // Put data to neighbors
     if (north != MPI_PROC_NULL) {
         // Our first row to north's last ghost row
         MPI_Put(&u[1][1], local_cols, MPI_DOUBLE, north, 
                 (local_rows + 1) * (local_cols + 2) + 1, local_cols, MPI_DOUBLE, win);
     }
     
     if (south != MPI_PROC_NULL) {
         // Our last row to south's first ghost row
         MPI_Put(&u[local_rows][1], local_cols, MPI_DOUBLE, south, 
                 1, local_cols, MPI_DOUBLE, win);
     }
     
     if (west != MPI_PROC_NULL) {
         // Our first column to west's last ghost column
         for (int i = 1; i <= local_rows; i++) {
             MPI_Put(&u[i][1], 1, MPI_DOUBLE, west, 
                    i * (local_cols + 2) + local_cols + 1, 1, MPI_DOUBLE, win);
         }
     }
     
     if (east != MPI_PROC_NULL) {
         // Our last column to east's first ghost column
         for (int i = 1; i <= local_rows; i++) {
             MPI_Put(&u[i][local_cols], 1, MPI_DOUBLE, east, 
                    i * (local_cols + 2), 1, MPI_DOUBLE, win);
         }
     }
     
     // End RMA epoch and synchronize
     MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
     
     // Copy data from win_mem back to u
     for (int i = 0; i <= local_rows + 1; i++) {
         for (int j = 0; j <= local_cols + 1; j++) {
             u[i][j] = win_mem[i * (local_cols + 2) + j];
         }
     }
 }
 
 /**
  * Exchange halos using RMA with general active target synchronization
  */
 void exchange_halos_rma_gats(double **u, int local_rows, int local_cols, MPI_Win win, double *win_mem, 
                            int *neighbors, int num_neighbors) {
     int local_size = (local_rows + 2) * (local_cols + 2);
     MPI_Group world_group, origin_group, target_group;
     
     // Create groups for origin (processes that will access the window) and
     // target (processes whose windows will be accessed)
     MPI_Comm_group(cart_comm, &world_group);
     
     // All neighbors whose data we need 
     int origin_neighbors[4] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
     int origin_count = 0;
     if (north != MPI_PROC_NULL) origin_neighbors[origin_count++] = north;
     if (south != MPI_PROC_NULL) origin_neighbors[origin_count++] = south;
     if (east != MPI_PROC_NULL) origin_neighbors[origin_count++] = east;
     if (west != MPI_PROC_NULL) origin_neighbors[origin_count++] = west;
     
     // All neighbors who need our data
     int target_neighbors[4] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
     int target_count = 0;
     if (north != MPI_PROC_NULL) target_neighbors[target_count++] = north;
     if (south != MPI_PROC_NULL) target_neighbors[target_count++] = south;
     if (east != MPI_PROC_NULL) target_neighbors[target_count++] = east;
     if (west != MPI_PROC_NULL) target_neighbors[target_count++] = west;
     
     // Create groups from neighbor lists
     MPI_Group_incl(world_group, origin_count, origin_neighbors, &origin_group);
     MPI_Group_incl(world_group, target_count, target_neighbors, &target_group);
     
     // Copy local array to window memory
     for (int i = 0; i <= local_rows + 1; i++) {
         for (int j = 0; j <= local_cols + 1; j++) {
             win_mem[i * (local_cols + 2) + j] = u[i][j];
         }
     }
     
     // Start exposure epoch with neighbors - allowing them to access our window
     MPI_Win_post(target_group, 0, win);
     
     // Start access epoch to neighbors - allowing us to access their windows
     MPI_Win_start(origin_group, 0, win);
     
     // Put data to neighbors
     if (north != MPI_PROC_NULL) {
         // Our first row to north's last ghost row
         MPI_Put(&u[1][1], local_cols, MPI_DOUBLE, north, 
                 (local_rows + 1) * (local_cols + 2) + 1, local_cols, MPI_DOUBLE, win);
     }
     
     if (south != MPI_PROC_NULL) {
         // Our last row to south's first ghost row
         MPI_Put(&u[local_rows][1], local_cols, MPI_DOUBLE, south, 
                 1, local_cols, MPI_DOUBLE, win);
     }
     
     if (west != MPI_PROC_NULL) {
         // Our first column to west's last ghost column
         for (int i = 1; i <= local_rows; i++) {
             MPI_Put(&u[i][1], 1, MPI_DOUBLE, west, 
                    i * (local_cols + 2) + local_cols + 1, 1, MPI_DOUBLE, win);
         }
     }
     
     if (east != MPI_PROC_NULL) {
         // Our last column to east's first ghost column
         for (int i = 1; i <= local_rows; i++) {
             MPI_Put(&u[i][local_cols], 1, MPI_DOUBLE, east, 
                    i * (local_cols + 2), 1, MPI_DOUBLE, win);
         }
     }
     
     // Complete access epoch
     MPI_Win_complete(win);
     
     // Complete exposure epoch
     MPI_Win_wait(win);
     
     // Copy data from win_mem back to u
     for (int i = 0; i <= local_rows + 1; i++) {
         for (int j = 0; j <= local_cols + 1; j++) {
             u[i][j] = win_mem[i * (local_cols + 2) + j];
         }
     }
     
     // Free groups
     MPI_Group_free(&origin_group);
     MPI_Group_free(&target_group);
     MPI_Group_free(&world_group);
 }
 
 /**
  * Copy array data
  */
 void copy_array(double **dst, double **src, int local_rows, int local_cols) {
     for (int i = 0; i <= local_rows + 1; i++) {
         for (int j = 0; j <= local_cols + 1; j++) {
             dst[i][j] = src[i][j];
         }
     }
 }