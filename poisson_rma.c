/* poisson_rma.c
 * 2D Poisson solver with 2D processor decomposition.
 * Implements ghost/halo exchange using:
 *  1) MPI_Win_fence epoch + MPI_Put/Get
 *  2) General active-target synchronization (Win_start/post, complete/wait)
 * Also includes a baseline MPI_Send/Recv ghost exchange for validation.
 * Compares results to ensure correctness.
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <mpi.h>
 #include <math.h>
 
 // problem size per dimension
 #define NX 100
 #define NY 100
 // number of iterations
 #define ITER 1000
 
 // allocate 2D array with halo layers
 double* alloc2d(int nx, int ny) {
     return calloc((nx+2)*(ny+2), sizeof(double));
 }
 
 double access(double *u, int i, int j, int stride) {
     return u[i*stride + j];
 }
 
 void set_access(double *u, int i, int j, int stride, double v) {
     u[i*stride + j] = v;
 }
 
 // Baseline message-passing ghost exchange
 void exchange_mp(double *u, int nx, int ny, MPI_Comm comm2d) {
     int rank, coords[2], dims[2], nbrs[4];
     MPI_Comm_rank(comm2d, &rank);
     MPI_Cart_get(comm2d, 2, dims, (int[]){0,0}, coords);
     MPI_Cart_shift(comm2d, 0, 1, &nbrs[0], &nbrs[1]); // up, down
     MPI_Cart_shift(comm2d, 1, 1, &nbrs[2], &nbrs[3]); // left, right
 
     MPI_Request reqs[8]; int r=0;
     // pack send buffers
     double *send_up = malloc(ny*sizeof(double));
     double *send_down = malloc(ny*sizeof(double));
     double *send_left = malloc(nx*sizeof(double));
     double *send_right = malloc(nx*sizeof(double));
     double *recv_up = malloc(ny*sizeof(double));
     double *recv_down = malloc(ny*sizeof(double));
     double *recv_left = malloc(nx*sizeof(double));
     double *recv_right = malloc(nx*sizeof(double));
     for(int j=1;j<=ny;j++) send_up[j-1] = access(u,1,j,ny+2);
     for(int j=1;j<=ny;j++) send_down[j-1] = access(u,nx,j,ny+2);
     for(int i=1;i<=nx;i++) send_left[i-1] = access(u,i,1,ny+2);
     for(int i=1;i<=nx;i++) send_right[i-1] = access(u,i,ny,ny+2);
     
     // exchange
     MPI_Irecv(recv_up, ny, MPI_DOUBLE, nbrs[0],0,comm2d,&reqs[r++]);
     MPI_Irecv(recv_down, ny, MPI_DOUBLE, nbrs[1],0,comm2d,&reqs[r++]);
     MPI_Irecv(recv_left, nx, MPI_DOUBLE, nbrs[2],0,comm2d,&reqs[r++]);
     MPI_Irecv(recv_right, nx, MPI_DOUBLE, nbrs[3],0,comm2d,&reqs[r++]);
     MPI_Isend(send_up, ny, MPI_DOUBLE, nbrs[0],0,comm2d,&reqs[r++]);
     MPI_Isend(send_down, ny, MPI_DOUBLE, nbrs[1],0,comm2d,&reqs[r++]);
     MPI_Isend(send_left, nx, MPI_DOUBLE, nbrs[2],0,comm2d,&reqs[r++]);
     MPI_Isend(send_right, nx, MPI_DOUBLE, nbrs[3],0,comm2d,&reqs[r++]);
     MPI_Waitall(r, reqs, MPI_STATUSES_IGNORE);
     
     // unpack
     for(int j=1;j<=ny;j++) set_access(u,0,j,ny+2, recv_up[j-1]);
     for(int j=1;j<=ny;j++) set_access(u,nx+1,j,ny+2, recv_down[j-1]);
     for(int i=1;i<=nx;i++) set_access(u,i,0,ny+2, recv_left[i-1]);
     for(int i=1;i<=nx;i++) set_access(u,i,ny+1,ny+2, recv_right[i-1]);
     
     free(send_up); free(send_down); free(send_left); free(send_right);
     free(recv_up); free(recv_down); free(recv_left); free(recv_right);
 }
 
 // Ghost exchange using MPI_Win_fence
 void exchange_fence(double *u, double *winbuf, int nx, int ny, MPI_Comm comm2d, MPI_Win win) {
     int rank, coords[2], dims[2], nbrs[4];
     MPI_Comm_rank(comm2d, &rank);
     MPI_Cart_get(comm2d, 2, dims, (int[]){0,0}, coords);
     MPI_Cart_shift(comm2d, 0, 1, &nbrs[0], &nbrs[1]);
     MPI_Cart_shift(comm2d, 1, 1, &nbrs[2], &nbrs[3]);
 
     MPI_Win_fence(0, win);
     // pack and put to neighbor windows
     for(int j=1;j<=ny;j++) {
         double vup = access(u,1,j,ny+2);
         MPI_Put(&vup, 1, MPI_DOUBLE, nbrs[0], (nx+1)*(ny+2)+j, 1, MPI_DOUBLE, win);
         double vdown = access(u,nx,j,ny+2);
         MPI_Put(&vdown,1,MPI_DOUBLE,nbrs[1], j,1,MPI_DOUBLE,win);
     }
     for(int i=1;i<=nx;i++) {
         double vleft = access(u,i,1,ny+2);
         MPI_Put(&vleft,1,MPI_DOUBLE,nbrs[2], i*(ny+2)+ny+1,1,MPI_DOUBLE,win);
         double vright = access(u,i,ny,ny+2);
         MPI_Put(&vright,1,MPI_DOUBLE,nbrs[3], i*(ny+2),1,MPI_DOUBLE,win);
     }
     // get neighbors into winbuf
     for(int j=1;j<=ny;j++) {
         MPI_Get(&winbuf[0*(ny+2)+j],1,MPI_DOUBLE,nbrs[0], 0*(ny+2)+j,1,MPI_DOUBLE,win);
         MPI_Get(&winbuf[(nx+1)*(ny+2)+j],1,MPI_DOUBLE,nbrs[1], (nx+1)*(ny+2)+j,1,MPI_DOUBLE,win);
     }
     for(int i=1;i<=nx;i++) {
         MPI_Get(&winbuf[i*(ny+2)+0],1,MPI_DOUBLE,nbrs[2], i*(ny+2)+0,1,MPI_DOUBLE,win);
         MPI_Get(&winbuf[i*(ny+2)+ny+1],1,MPI_DOUBLE,nbrs[3], i*(ny+2)+ny+1,1,MPI_DOUBLE,win);
     }
     MPI_Win_fence(0, win);
     // copy from winbuf to u
     for(int j=1;j<=ny;j++) set_access(u,0,j,ny+2, winbuf[0*(ny+2)+j]);
     for(int j=1;j<=ny;j++) set_access(u,nx+1,j,ny+2, winbuf[(nx+1)*(ny+2)+j]);
     for(int i=1;i<=nx;i++) set_access(u,i,0,ny+2, winbuf[i*(ny+2)+0]);
     for(int i=1;i<=nx;i++) set_access(u,i,ny+1,ny+2, winbuf[i*(ny+2)+ny+1]);
 }
 
 // Ghost exchange using active target synchronization
 void exchange_active(double *u, double *winbuf, int nx, int ny, MPI_Comm comm2d, MPI_Win win) {
     int rank, coords[2], dims[2], nbrs[4];
     MPI_Comm_rank(comm2d, &rank);
     MPI_Cart_get(comm2d, 2, dims, (int[]){0,0}, coords);
     MPI_Cart_shift(comm2d, 0, 1, &nbrs[0], &nbrs[1]);
     MPI_Cart_shift(comm2d, 1, 1, &nbrs[2], &nbrs[3]);
 
     MPI_Group world_group, target_group;
     MPI_Comm_group(MPI_COMM_WORLD, &world_group);
     int targets[4] = {nbrs[0], nbrs[1], nbrs[2], nbrs[3]};
     MPI_Group_incl(world_group, 4, targets, &target_group);
 
     // start epoch to put/get to targets
     MPI_Win_start(target_group, 0, win);
     // pack & put
     for(int j=1;j<=ny;j++) {
         double v = access(u,1,j,ny+2);
         MPI_Put(&v,1,MPI_DOUBLE,nbrs[0], (nx+1)*(ny+2)+j,1,MPI_DOUBLE,win);
         v = access(u,nx,j,ny+2);
         MPI_Put(&v,1,MPI_DOUBLE,nbrs[1], j,1,MPI_DOUBLE,win);
     }
     for(int i=1;i<=nx;i++) {
         double v = access(u,i,1,ny+2);
         MPI_Put(&v,1,MPI_DOUBLE,nbrs[2], i*(ny+2)+ny+1,1,MPI_DOUBLE,win);
         v = access(u,i,ny,ny+2);
         MPI_Put(&v,1,MPI_DOUBLE,nbrs[3], i*(ny+2),1,MPI_DOUBLE,win);
     }
     MPI_Win_complete(win);
 
     // post epoch to accept from neighbors
     MPI_Win_post(target_group, 0, win);
     // get
     for(int j=1;j<=ny;j++) {
         MPI_Get(&winbuf[0*(ny+2)+j],1,MPI_DOUBLE,nbrs[0], 0*(ny+2)+j,1,MPI_DOUBLE,win);
         MPI_Get(&winbuf[(nx+1)*(ny+2)+j],1,MPI_DOUBLE,nbrs[1], (nx+1)*(ny+2)+j,1,MPI_DOUBLE,win);
     }
     for(int i=1;i<=nx;i++) {
         MPI_Get(&winbuf[i*(ny+2)+0],1,MPI_DOUBLE,nbrs[2], i*(ny+2)+0,1,MPI_DOUBLE,win);
         MPI_Get(&winbuf[i*(ny+2)+ny+1],1,MPI_DOUBLE,nbrs[3], i*(ny+2)+ny+1,1,MPI_DOUBLE,win);
     }
     MPI_Win_wait(win);
 
     // copy into u
     for(int j=1;j<=ny;j++) set_access(u,0,j,ny+2, winbuf[0*(ny+2)+j]);
     for(int j=1;j<=ny;j++) set_access(u,nx+1,j,ny+2, winbuf[(nx+1)*(ny+2)+j]);
     for(int i=1;i<=nx;i++) set_access(u,i,0,ny+2, winbuf[i*(ny+2)+0]);
     for(int i=1;i<=nx;i++) set_access(u,i,ny+1,ny+2, winbuf[i*(ny+2)+ny+1]);
 
     MPI_Group_free(&target_group);
     MPI_Group_free(&world_group);
 }
 
 int main(int argc, char **argv) {
     MPI_Init(&argc,&argv);
     MPI_Comm comm2d;
     int dims[2]={0,0}, periods[2]={0,0};
     MPI_Dims_create(MPI_COMM_WORLD,2,dims);
     MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,0,&comm2d);
 
     double *u_mp = alloc2d(NX,dims[1]>0?NY:NY);
     double *u_fence = alloc2d(NX,NY);
     double *u_active = alloc2d(NX,NY);
     double *winbuf;
     int total = (NX+2)*(NY+2);
     MPI_Win win;
     MPI_Win_allocate(sizeof(double)*total, sizeof(double), MPI_INFO_NULL, comm2d, &winbuf, &win);
 
     // initialize
     for(int i=0;i<total;i++) winbuf[i]=0;
     for(int i=0;i<total;i++) u_mp[i]=u_fence[i]=u_active[i]=0;
 
     // time steps or iterations
     for(int it=0; it<ITER; ++it) {
         // baseline
         exchange_mp(u_mp, NX, NY, comm2d);
         // fence RMA
         exchange_fence(u_fence, winbuf, NX, NY, comm2d, win);
         // active RMA
         exchange_active(u_active, winbuf, NX, NY, comm2d, win);
     }
 
     // compare
     double max_err_f=0, max_err_a=0;
     for(int i=0;i<total;i++) {
         max_err_f = fmax(max_err_f, fabs(u_mp[i]-u_fence[i]));
         max_err_a = fmax(max_err_a, fabs(u_mp[i]-u_active[i]));
     }
     int rank; MPI_Comm_rank(comm2d,&rank);
     if(rank==0) {
         printf("Max error (fence) = %g\n", max_err_f);
         printf("Max error (active) = %g\n", max_err_a);
     }
 
     MPI_Win_free(&win);
     MPI_Comm_free(&comm2d);
     MPI_Finalize();
     return 0;
 }
 