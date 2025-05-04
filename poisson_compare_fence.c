/*
 * poisson_compare_fence.c
 *
 * Solve the 2D Poisson equation on [0,1]×[0,1] with homogeneous RHS
 * and Dirichlet boundary u=sin(pi*x) on y=0, u=0 on the other three sides.
 * Use Jacobi iteration to tolerance, once with MPI_Sendrecv halo‐exchange,
 * once with one‐sided RMA + MPI_Win_fence.  Finally compute L2 difference.
 *
 * Usage:
 *   mpirun -np P ./poisson_compare_fence nx ny tol maxiter
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <mpi.h>
 
 static inline int idx(int i, int j, int nx, int ny){ return i*(ny+2)+j; }
 
 /* Initialize boundary conditions into u[0..nx+1][0..ny+1] */
 void init_bc(double *u, int nx, int ny, double h){
   for(int i=0;i<=nx+1;i++) for(int j=0;j<=ny+1;j++) u[idx(i,j,nx,ny)] = 0.0;
   /* y=0 boundary: u(x,0)=sin(pi*x) */
   for(int i=0;i<=nx+1;i++){
     double x = i*h;
     u[idx(i,0,nx,ny)] = sin(M_PI*x);
   }
 }
 
 /* Jacobi with message‐passing halo exchange */
 void jacobi_msg(double *u, double *u_new, int nx, int ny,
                 int north,int south,int east,int west,
                 MPI_Comm comm, double tol, int maxiter){
   MPI_Request reqs[8];
   int iter=0; double diff;
   double *tmp;
   double h=1.0/nx;
 
   do {
     /* exchange halos */
     /* send up, recv from down */
     MPI_Isend(&u[idx(1,1,nx,ny)], ny, MPI_DOUBLE, north,0,comm,&reqs[0]);
     MPI_Irecv(&u[idx(1,ny+1,nx,ny)],ny,MPI_DOUBLE,south,0,comm,&reqs[1]);
     /* send down, recv from up */
     MPI_Isend(&u[idx(1,ny,nx,ny)],ny,MPI_DOUBLE,south,1,comm,&reqs[2]);
     MPI_Irecv(&u[idx(1,0,nx,ny)],ny,MPI_DOUBLE,north,1,comm,&reqs[3]);
     /* send right, recv from left */
     MPI_Isend(&u[idx(1,1,nx,ny)],1,MPI_DOUBLE,east,2,comm,&reqs[4]);
     MPI_Irecv(&u[idx(0,1,nx,ny)],1,MPI_DOUBLE,west,2,comm,&reqs[5]);
     /* send left, recv from right */
     MPI_Isend(&u[idx(nx,1,nx,ny)],1,MPI_DOUBLE,west,3,comm,&reqs[6]);
     MPI_Irecv(&u[idx(nx+1,1,nx,ny)],1,MPI_DOUBLE,east,3,comm,&reqs[7]);
     MPI_Waitall(8,reqs,MPI_STATUSES_IGNORE);
 
     /* Jacobi update */
     diff = 0.0;
     for(int i=1;i<=nx;i++){
       for(int j=1;j<=ny;j++){
         double v = 0.25*( u[idx(i+1,j,nx,ny)] + u[idx(i-1,j,nx,ny)]
                         +u[idx(i,j+1,nx,ny)] + u[idx(i,j-1,nx,ny)] );
         u_new[idx(i,j,nx,ny)] = v;
         diff = fmax(diff, fabs(v - u[idx(i,j,nx,ny)]));
       }
     }
     /* swap and check */
     for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) u[idx(i,j,nx,ny)] = u_new[idx(i,j,nx,ny)];
     MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_MAX, comm);
     iter++;
   } while(diff>tol && iter<maxiter);
 }
 
 /* Halo‐exchange via one‐sided RMA + MPI_Win_fence */
 void exchange_fence(MPI_Win win, double *u, int nx,int ny,
                     int north,int south,int east,int west,
                     MPI_Datatype coltype){
   MPI_Win_fence(0,win);
   /* East → right neighbor’s left ghost */
   if(east!=MPI_PROC_NULL)
     MPI_Put(&u[idx(nx,1,nx,ny)],1,coltype,east,
             /* disp = index of (0,1) */
             0*(ny+2)+1,1,coltype,win);
   /* West → left neighbor’s right ghost */
   if(west!=MPI_PROC_NULL)
     MPI_Put(&u[idx(1,1,nx,ny)],1,coltype,west,
             (nx+1)*(ny+2)+1,1,coltype,win);
   /* North → top neighbor’s bottom ghost */
   if(north!=MPI_PROC_NULL)
     MPI_Put(&u[idx(1,ny,nx,ny)],nx,MPI_DOUBLE,north,
             1*(ny+2)+0,nx,MPI_DOUBLE,win);
   /* South → bottom neighbor’s top ghost */
   if(south!=MPI_PROC_NULL)
     MPI_Put(&u[idx(1,1,nx,ny)],nx,MPI_DOUBLE,south,
             1*(ny+2)+(ny+1),nx,MPI_DOUBLE,win);
   MPI_Win_fence(0,win);
 }
 
 /* Jacobi with RMA fence exchange */
 void jacobi_fence(double *u, double *u_new, int nx,int ny,
                   int north,int south,int east,int west,
                   MPI_Win win, MPI_Datatype coltype,
                   MPI_Comm comm, double tol, int maxiter){
   int iter=0; double diff;
   do {
     exchange_fence(win,u,nx,ny,north,south,east,west,coltype);
     /* same update as before */
     diff=0.0;
     for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++){
       double v=0.25*(u[idx(i+1,j,nx,ny)]+u[idx(i-1,j,nx,ny)]
                     +u[idx(i,j+1,nx,ny)]+u[idx(i,j-1,nx,ny)]);
       u_new[idx(i,j,nx,ny)]=v;
       diff=fmax(diff,fabs(v-u[idx(i,j,nx,ny)]));
     }
     for(int i=1;i<=nx;i++) for(int j=1;j<=ny;j++) u[idx(i,j,nx,ny)]=u_new[idx(i,j,nx,ny)];
     MPI_Allreduce(MPI_IN_PLACE,&diff,1,MPI_DOUBLE,MPI_MAX,comm);
     iter++;
   } while(diff>tol && iter<maxiter);
 }
 
 int main(int ac, char **av){
   if(ac<5){ if(!MPI_Initialized(NULL)) MPI_Init(&ac,&av);
     if(!MPI_Comm_rank(MPI_COMM_WORLD,&ac)) 
       fprintf(stderr,"Usage: mpirun -np P %s nx ny tol maxiter\n",av[0]);
     MPI_Abort(MPI_COMM_WORLD,1);
   }
   int nx=atoi(av[1]), ny=atoi(av[2]);
   double tol=atof(av[3]); int maxiter=atoi(av[4]);
 
   MPI_Init(&ac,&av);
   MPI_Comm comm = MPI_COMM_WORLD;
   int rank, P; MPI_Comm_size(comm,&P); MPI_Comm_rank(comm,&rank);
 
   /* set up 2D Cartesian */
   int dims[2]={0,0}, periods[2]={0,0}, coords[2];
   MPI_Dims_create(P,2,dims);
   MPI_Cart_create(comm,2,dims,periods,0,&comm);
   MPI_Cart_coords(comm,rank,2,coords);
   int west,north,east,south;
   MPI_Cart_shift(comm,0,1,&west,&east);
   MPI_Cart_shift(comm,1,1,&south,&north);
 
   /* local arrays */
   double h=1.0/nx;
   int L=(nx+2)*(ny+2);
   double *u_msg   = calloc(L,sizeof(double));
   double *u_msg_n = calloc(L,sizeof(double));
   double *u_rma   = calloc(L,sizeof(double));
   double *u_rma_n = calloc(L,sizeof(double));
 
   /* init BC for both */
   init_bc(u_msg,nx,ny,h);
   init_bc(u_rma,nx,ny,h);
 
   /* message‐passing solve */
   jacobi_msg(u_msg,u_msg_n,nx,ny,north,south,east,west,comm,tol,maxiter);
 
   /* create RMA window */
   MPI_Datatype coltype;
   MPI_Type_vector(ny,1,ny+2,MPI_DOUBLE,&coltype);
   MPI_Type_commit(&coltype);
   MPI_Win win;
   MPI_Win_create(u_rma,L,sizeof(double),MPI_INFO_NULL,comm,&win);
 
   /* RMA solve */
   jacobi_fence(u_rma,u_rma_n,nx,ny,north,south,east,west,win,coltype,comm,tol,maxiter);
 
   /* compare */
   double loc_err2=0.0;
   for(int i=1;i<=nx;i++)for(int j=1;j<=ny;j++){
     double d = u_msg[idx(i,j,nx,ny)]-u_rma[idx(i,j,nx,ny)];
     loc_err2 += d*d;
   }
   double err2;
   MPI_Allreduce(&loc_err2,&err2,1,MPI_DOUBLE,MPI_SUM,comm);
   double err = sqrt(err2);
   if(rank==0) printf("L2 difference (msg vs fence): %e\n", err);
 
   MPI_Win_free(&win);
   MPI_Type_free(&coltype);
   free(u_msg); free(u_msg_n);
   free(u_rma); free(u_rma_n);
 
   MPI_Finalize();
   return 0;
 }
 