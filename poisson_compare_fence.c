/*
 * poisson_compare_fence.c
 *
 * 2D Poisson solve (Jacobi) on [0,1]^2, Dirichlet BC sin(pi x) at y=0,
 * homogeneous elsewhere. Compare MPI_Sendrecv vs. one-sided RMA (Fence).
 *
 * Usage:
 *   mpirun -np P ./poisson_compare_fence nx ny tol maxiter
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <mpi.h>
 
 static inline int idx(int i,int j,int nx,int ny){ return i*(ny+2)+j; }
 
 /* Initialize BC: u(x,0)=sin(pi x), rest zero */
 void init_bc(double *u, int nx, int ny, double h) {
     for(int i=0;i<=nx+1;i++)
         for(int j=0;j<=ny+1;j++)
             u[idx(i,j,nx,ny)] = 0.0;
     for(int i=0;i<=nx+1;i++){
         double x = i*h;
         u[idx(i,0,nx,ny)] = sin(M_PI*x);
     }
 }
 
 /* Jacobi with MPI_Sendrecv halo-exchange */
 void jacobi_msg(double *u, double *u_new,
                 int nx,int ny,
                 int north,int south,int east,int west,
                 MPI_Comm comm,
                 double tol,int maxiter)
 {
     MPI_Request reqs[8];
     double diff;
     int iter=0;
     do {
         MPI_Isend(&u[idx(1,1,nx,ny)], ny, MPI_DOUBLE, north, 0, comm, &reqs[0]);
         MPI_Irecv(&u[idx(1,ny+1,nx,ny)], ny, MPI_DOUBLE, south, 0, comm, &reqs[1]);
         MPI_Isend(&u[idx(1,ny,nx,ny)], ny, MPI_DOUBLE, south, 1, comm, &reqs[2]);
         MPI_Irecv(&u[idx(1,0,nx,ny)],  ny, MPI_DOUBLE, north, 1, comm, &reqs[3]);
         MPI_Isend(&u[idx(1,1,nx,ny)], 1, MPI_DOUBLE, east,  2, comm, &reqs[4]);
         MPI_Irecv(&u[idx(0,1,nx,ny)],  1, MPI_DOUBLE, west,  2, comm, &reqs[5]);
         MPI_Isend(&u[idx(nx,1,nx,ny)], 1, MPI_DOUBLE, west,  3, comm, &reqs[6]);
         MPI_Irecv(&u[idx(nx+1,1,nx,ny)],1, MPI_DOUBLE, east,  3, comm, &reqs[7]);
         MPI_Waitall(8, reqs, MPI_STATUSES_IGNORE);
 
         diff = 0.0;
         for(int i=1;i<=nx;i++){
             for(int j=1;j<=ny;j++){
                 double v = 0.25*(u[idx(i+1,j,nx,ny)] + u[idx(i-1,j,nx,ny)]
                                + u[idx(i,j+1,nx,ny)] + u[idx(i,j-1,nx,ny)]);
                 u_new[idx(i,j,nx,ny)] = v;
                 diff = fmax(diff, fabs(v - u[idx(i,j,nx,ny)]));
             }
         }
         for(int i=1;i<=nx;i++)
             for(int j=1;j<=ny;j++)
                 u[idx(i,j,nx,ny)] = u_new[idx(i,j,nx,ny)];
 
         MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, comm);
         iter++;
     } while(diff>tol && iter<maxiter);
 }
 
 /* RMA fence-based ghost exchange, using both column and row types */
 void exchange_fence(MPI_Win win,
                     double *u,
                     int nx,int ny,
                     int north,int south,int east,int west,
                     MPI_Datatype coltype, MPI_Datatype rowtype)
 {
     MPI_Win_fence(0, win);
 
     if(east!=MPI_PROC_NULL) {
         MPI_Put(&u[idx(nx,1,nx,ny)], 1, coltype,
                 east, idx(0,1,nx,ny), 1, coltype, win);
     }
     if(west!=MPI_PROC_NULL) {
         MPI_Put(&u[idx(1,1,nx,ny)], 1, coltype,
                 west, idx(nx+1,1,nx,ny), 1, coltype, win);
     }
     if(north!=MPI_PROC_NULL) {
         MPI_Put(&u[idx(1,ny,nx,ny)], 1, rowtype,
                 north, idx(1,0,nx,ny), 1, rowtype, win);
     }
     if(south!=MPI_PROC_NULL) {
         MPI_Put(&u[idx(1,1,nx,ny)], 1, rowtype,
                 south, idx(1,ny+1,nx,ny), 1, rowtype, win);
     }
 
     MPI_Win_fence(0, win);
 }
 
 /* Jacobi with RMA fence exchange */
 void jacobi_fence(double *u, double *u_new,
                   int nx,int ny,
                   int north,int south,int east,int west,
                   MPI_Win win,
                   MPI_Datatype coltype, MPI_Datatype rowtype,
                   MPI_Comm comm,
                   double tol,int maxiter)
 {
     int iter=0;
     double diff;
     do {
         exchange_fence(win, u, nx, ny, north, south, east, west, coltype, rowtype);
         diff = 0.0;
         for(int i=1;i<=nx;i++){
             for(int j=1;j<=ny;j++){
                 double v = 0.25*(u[idx(i+1,j,nx,ny)] + u[idx(i-1,j,nx,ny)]
                                + u[idx(i,j+1,nx,ny)] + u[idx(i,j-1,nx,ny)]);
                 u_new[idx(i,j,nx,ny)] = v;
                 diff = fmax(diff, fabs(v - u[idx(i,j,nx,ny)]));
             }
         }
         for(int i=1;i<=nx;i++)
             for(int j=1;j<=ny;j++)
                 u[idx(i,j,nx,ny)] = u_new[idx(i,j,nx,ny)];
         MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, comm);
         iter++;
     } while(diff>tol && iter<maxiter);
 }
 
 int main(int ac, char **av) {
     if(ac<5) {
         fprintf(stderr, "Usage: %s nx ny tol maxiter\n", av[0]);
         MPI_Abort(MPI_COMM_WORLD,1);
     }
 
     MPI_Init(&ac, &av);
     int nx = atoi(av[1]), ny = atoi(av[2]);
     double tol = atof(av[3]);
     int maxiter = atoi(av[4]);
 
     MPI_Comm comm = MPI_COMM_WORLD;
     int P, rank;
     MPI_Comm_size(comm, &P);
     MPI_Comm_rank(comm,&rank);
 
     int dims[2]={0,0}, periods[2]={0,0}, coords[2];
     MPI_Dims_create(P,2,dims);
     MPI_Cart_create(comm,2,dims,periods,0,&comm);
     MPI_Cart_coords(comm, rank,2,coords);
     int west,north,east,south;
     MPI_Cart_shift(comm,0,1,&west,&east);
     MPI_Cart_shift(comm,1,1,&south,&north);
 
     int L = (nx+2)*(ny+2);
     double *u_msg   = calloc(L,sizeof(double));
     double *u_msg_n = calloc(L,sizeof(double));
     double *u_rma   = calloc(L,sizeof(double));
     double *u_rma_n = calloc(L,sizeof(double));
 
     double h = 1.0 / nx;
     init_bc(u_msg, nx, ny, h);
     init_bc(u_rma, nx, ny, h);
 
     jacobi_msg(u_msg, u_msg_n, nx, ny,
                north, south, east, west,
                comm, tol, maxiter);
 
     MPI_Datatype coltype, rowtype;
     MPI_Type_vector(ny, 1, ny+2, MPI_DOUBLE, &coltype);
     MPI_Type_commit(&coltype);
     MPI_Type_vector(nx, 1, ny+2, MPI_DOUBLE, &rowtype);
     MPI_Type_commit(&rowtype);
 
     MPI_Win win;
     MPI_Win_create(u_rma, L, sizeof(double), MPI_INFO_NULL, comm, &win);
 
     jacobi_fence(u_rma, u_rma_n, nx, ny,
                  north, south, east, west,
                  win, coltype, rowtype,
                  comm, tol, maxiter);
 
     double loc2=0.0;
     for(int i=1;i<=nx;i++)
         for(int j=1;j<=ny;j++){
             double d = u_msg[idx(i,j,nx,ny)] - u_rma[idx(i,j,nx,ny)];
             loc2 += d*d;
         }
     double sum2;
     MPI_Allreduce(&loc2, &sum2, 1, MPI_DOUBLE, MPI_SUM, comm);
     if(rank==0)
         printf("L2 difference (msg vs fence): %e\n", sqrt(sum2));
 
     MPI_Win_free(&win);
     MPI_Type_free(&coltype);
     MPI_Type_free(&rowtype);
     free(u_msg); free(u_msg_n);
     free(u_rma); free(u_rma_n);
 
     MPI_Finalize();
     return 0;
 }
 