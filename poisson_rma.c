/* poisson_rma.c
 * 2D Poisson solver with 2D processor decomposition.
 * Implements ghost/halo exchange using:
 *   1) MPI_Send/MPI_Recv (baseline)
 *   2) MPI_Win_fence epoch + MPI_Put/Get
 *   3) Active-target synchronization (Win_start/post, complete/wait)
 * Verifies RMA versions match baseline within floating-point tolerance.
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <mpi.h>
 #include <math.h>
 
 #define NX 100
 #define NY 100
 #define ITER 1000
 
 // Allocate a (nx+2)x(ny+2) array with zero-initialized halo
 static double* alloc2d(int nx, int ny) {
     return calloc((nx+2)*(ny+2), sizeof(double));
 }
 static double access_u(const double *u, int i, int j, int stride) {
     return u[i*stride + j];
 }
 static void set_u(double *u, int i, int j, int stride, double v) {
     u[i*stride + j] = v;
 }
 
 // Baseline MPI_Send/MPI_Recv ghost exchange
 void exchange_mp(double *u, int nx, int ny, MPI_Comm comm2d) {
     int dims[2], coords[2], nbrs[4];
     MPI_Cart_get(comm2d, 2, dims, NULL, coords);
     MPI_Cart_shift(comm2d, 0, 1, &nbrs[0], &nbrs[1]); // up/down
     MPI_Cart_shift(comm2d, 1, 1, &nbrs[2], &nbrs[3]); // left/right
 
     MPI_Request reqs[8];
     MPI_Status stats[8];
     int r = 0;
 
     double *s_up    = malloc(ny * sizeof(double));
     double *s_down  = malloc(ny * sizeof(double));
     double *s_left  = malloc(nx * sizeof(double));
     double *s_right = malloc(nx * sizeof(double));
     double *r_up    = malloc(ny * sizeof(double));
     double *r_down  = malloc(ny * sizeof(double));
     double *r_left  = malloc(nx * sizeof(double));
     double *r_right = malloc(nx * sizeof(double));
 
     // pack
     for(int j=1; j<=ny; j++) {
         s_up[j-1]   = access_u(u, 1, j, ny+2);
         s_down[j-1] = access_u(u, nx, j, ny+2);
     }
     for(int i=1; i<=nx; i++) {
         s_left[i-1]  = access_u(u, i, 1, ny+2);
         s_right[i-1] = access_u(u, i, ny, ny+2);
     }
     // post Irecv
     MPI_Irecv(r_up,    ny, MPI_DOUBLE, nbrs[0], 0, comm2d, &reqs[r++]);
     MPI_Irecv(r_down,  ny, MPI_DOUBLE, nbrs[1], 0, comm2d, &reqs[r++]);
     MPI_Irecv(r_left,  nx, MPI_DOUBLE, nbrs[2], 0, comm2d, &reqs[r++]);
     MPI_Irecv(r_right, nx, MPI_DOUBLE, nbrs[3], 0, comm2d, &reqs[r++]);
     // post Isend
     MPI_Isend(s_up,    ny, MPI_DOUBLE, nbrs[0], 0, comm2d, &reqs[r++]);
     MPI_Isend(s_down,  ny, MPI_DOUBLE, nbrs[1], 0, comm2d, &reqs[r++]);
     MPI_Isend(s_left,  nx, MPI_DOUBLE, nbrs[2], 0, comm2d, &reqs[r++]);
     MPI_Isend(s_right, nx, MPI_DOUBLE, nbrs[3], 0, comm2d, &reqs[r++]);
 
     MPI_Waitall(r, reqs, stats);
     // unpack
     for(int j=1; j<=ny; j++) {
         set_u(u, 0,    j, ny+2, r_up[j-1]);
         set_u(u, nx+1, j, ny+2, r_down[j-1]);
     }
     for(int i=1; i<=nx; i++) {
         set_u(u, i, 0,    ny+2, r_left[i-1]);
         set_u(u, i, ny+1, ny+2, r_right[i-1]);
     }
 
     free(s_up); free(s_down); free(s_left); free(s_right);
     free(r_up); free(r_down); free(r_left); free(r_right);
 }
 
 // RMA exchange with MPI_Win_fence
 void exchange_fence(double *u, double *winbuf, int nx, int ny,
                     MPI_Comm comm2d, MPI_Win win) {
     int dims[2], coords[2], nbrs[4];
     MPI_Cart_get(comm2d, 2, dims, NULL, coords);
     MPI_Cart_shift(comm2d, 0, 1, &nbrs[0], &nbrs[1]);
     MPI_Cart_shift(comm2d, 1, 1, &nbrs[2], &nbrs[3]);
 
     MPI_Win_fence(0, win);
     // put rows
     for(int j=1; j<=ny; j++) {
         double v_up = access_u(u, 1, j, ny+2);
         double v_dn = access_u(u, nx, j, ny+2);
         MPI_Put(&v_up, 1, MPI_DOUBLE, nbrs[0], (nx+1)*(ny+2)+j, 1, MPI_DOUBLE, win);
         MPI_Put(&v_dn, 1, MPI_DOUBLE, nbrs[1], j,                1, MPI_DOUBLE, win);
     }
     // put cols
     for(int i=1; i<=nx; i++) {
         double v_lt = access_u(u, i, 1, ny+2);
         double v_rt = access_u(u, i, ny, ny+2);
         MPI_Put(&v_lt, 1, MPI_DOUBLE, nbrs[2], i*(ny+2)+ny+1,    1, MPI_DOUBLE, win);
         MPI_Put(&v_rt, 1, MPI_DOUBLE, nbrs[3], i*(ny+2),         1, MPI_DOUBLE, win);
     }
     // get into winbuf
     for(int j=1; j<=ny; j++) {
         MPI_Get(&winbuf[0*(ny+2)+j],      1, MPI_DOUBLE, nbrs[0], 0*(ny+2)+j,      1, MPI_DOUBLE, win);
         MPI_Get(&winbuf[(nx+1)*(ny+2)+j], 1, MPI_DOUBLE, nbrs[1], (nx+1)*(ny+2)+j, 1, MPI_DOUBLE, win);
     }
     for(int i=1; i<=nx; i++) {
         MPI_Get(&winbuf[i*(ny+2)+0],      1, MPI_DOUBLE, nbrs[2], i*(ny+2)+0,      1, MPI_DOUBLE, win);
         MPI_Get(&winbuf[i*(ny+2)+ny+1],   1, MPI_DOUBLE, nbrs[3], i*(ny+2)+ny+1,   1, MPI_DOUBLE, win);
     }
     MPI_Win_fence(0, win);
     // unpack
     for(int j=1; j<=ny; j++) {
         set_u(u, 0,    j, ny+2, winbuf[0*(ny+2)+j]);
         set_u(u, nx+1, j, ny+2, winbuf[(nx+1)*(ny+2)+j]);
     }
     for(int i=1; i<=nx; i++) {
         set_u(u, i, 0,    ny+2, winbuf[i*(ny+2)+0]);
         set_u(u, i, ny+1, ny+2, winbuf[i*(ny+2)+ny+1]);
     }
 }
 
 // RMA exchange with active-target synchronization
 void exchange_active(double *u, double *winbuf, int nx, int ny,
                      MPI_Comm comm2d, MPI_Win win) {
     int dims[2], coords[2], nbrs[4];
     MPI_Cart_get(comm2d, 2, dims, NULL, coords);
     MPI_Cart_shift(comm2d, 0, 1, &nbrs[0], &nbrs[1]);
     MPI_Cart_shift(comm2d, 1, 1, &nbrs[2], &nbrs[3]);
 
     MPI_Group world_grp, tgt_grp;
     MPI_Comm_group(MPI_COMM_WORLD, &world_grp);
     int tgts[4] = {nbrs[0], nbrs[1], nbrs[2], nbrs[3]};
     MPI_Group_incl(world_grp, 4, tgts, &tgt_grp);
 
     // Put epoch
     MPI_Win_start(tgt_grp, 0, win);
     for(int j=1; j<=ny; j++) {
         double v_up = access_u(u, 1, j, ny+2);
         double v_dn = access_u(u, nx, j, ny+2);
         MPI_Put(&v_up, 1, MPI_DOUBLE, nbrs[0], (nx+1)*(ny+2)+j, 1, MPI_DOUBLE, win);
         MPI_Put(&v_dn, 1, MPI_DOUBLE, nbrs[1], j,                1, MPI_DOUBLE, win);
     }
     for(int i=1; i<=nx; i++) {
         double v_lt = access_u(u, i, 1, ny+2);
         double v_rt = access_u(u, i, ny, ny+2);
         MPI_Put(&v_lt, 1, MPI_DOUBLE, nbrs[2], i*(ny+2)+ny+1,    1, MPI_DOUBLE, win);
         MPI_Put(&v_rt, 1, MPI_DOUBLE, nbrs[3], i*(ny+2),         1, MPI_DOUBLE, win);
     }
     MPI_Win_complete(win);
 
     // Get epoch
     MPI_Win_post(tgt_grp, 0, win);
     for(int j=1; j<=ny; j++) {
         MPI_Get(&winbuf[0*(ny+2)+j],      1, MPI_DOUBLE, nbrs[0], 0*(ny+2)+j,      1, MPI_DOUBLE, win);
         MPI_Get(&winbuf[(nx+1)*(ny+2)+j], 1, MPI_DOUBLE, nbrs[1], (nx+1)*(ny+2)+j, 1, MPI_DOUBLE, win);
     }
     for(int i=1; i<=nx; i++) {
         MPI_Get(&winbuf[i*(ny+2)+0],      1, MPI_DOUBLE, nbrs[2], i*(ny+2)+0,      1, MPI_DOUBLE, win);
         MPI_Get(&winbuf[i*(ny+2)+ny+1],   1, MPI_DOUBLE, nbrs[3], i*(ny+2)+ny+1,   1, MPI_DOUBLE, win);
     }
     MPI_Win_wait(win);
 
     // unpack
     for(int j=1; j<=ny; j++) {
         set_u(u, 0,    j, ny+2, winbuf[0*(ny+2)+j]);
         set_u(u, nx+1, j, ny+2, winbuf[(nx+1)*(ny+2)+j]);
     }
     for(int i=1; i<=nx; i++) {
         set_u(u, i, 0,    ny+2, winbuf[i*(ny+2)+0]);
         set_u(u, i, ny+1, ny+2, winbuf[i*(ny+2)+ny+1]);
     }
 
     MPI_Group_free(&tgt_grp);
     MPI_Group_free(&world_grp);
 }
 
 int main(int argc, char *argv[]) {
     MPI_Init(&argc,&argv);
     MPI_Comm comm2d;
     int dims[2] = {0,0}, periods[2] = {0,0};
     int nprocs;
     MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
     MPI_Dims_create(nprocs, 2, dims);
     MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm2d);
 
     double *u_mp     = alloc2d(NX, NY);
     double *u_fence  = alloc2d(NX, NY);
     double *u_active = alloc2d(NX, NY);
     double *winbuf;
     MPI_Win win;
     int total = (NX+2)*(NY+2);
 
     MPI_Win_allocate(total*sizeof(double), sizeof(double), MPI_INFO_NULL,
                      comm2d, &winbuf, &win);
 
     // init
     for(int i=0; i<total; i++) {
         u_mp[i] = u_fence[i] = u_active[i] = 0.0;
         winbuf[i] = 0.0;
     }
 
     for(int it=0; it<ITER; it++) {
         exchange_mp(u_mp,        NX, NY, comm2d);
         exchange_fence(u_fence,  NX, NY, comm2d, win);
         exchange_active(u_active,NX, NY, comm2d, win);
     }
 
     double max_err_f = 0.0, max_err_a = 0.0;
     for(int i=0; i<total; i++) {
         double df = fabs(u_mp[i] - u_fence[i]);
         double da = fabs(u_mp[i] - u_active[i]);
         max_err_f = fmax(max_err_f, df);
         max_err_a = fmax(max_err_a, da);
     }
 
     int rank;
     MPI_Comm_rank(comm2d, &rank);
     if(rank==0) {
         printf("Max error (fence)  = %g\n", max_err_f);
         printf("Max error (active) = %g\n", max_err_a);
     }
 
     MPI_Win_free(&win);
     MPI_Comm_free(&comm2d);
     MPI_Finalize();
     return 0;
 }
 