# Poisson Solver with RMA-based Ghost Exchange

This project implements a 2D Poisson equation solver using a 5-point stencil finite difference method with a 2D processor decomposition. The key feature is the implementation of ghost/halo exchanges using MPI's Remote Memory Access (RMA) operations instead of traditional message passing.

## Features

- 2D domain decomposition for parallel processing
- Three ghost/halo exchange methods:
  1. Traditional MPI message passing (Send/Recv)
  2. RMA with MPI_Win_fence synchronization
  3. RMA with general active target synchronization (MPI_Win_start, MPI_Win_complete, MPI_Win_post, MPI_Win_wait)
- Jacobi iteration for solving the Poisson equation
- Performance timing for comparing different communication methods

## Compiling

To compile the program:

```bash
make
```

Or manually with:

```bash
mpicc -o poisson_solver_rma poisson_solver_rma.c -lm
```

## Usage

Run the program with:

```bash
mpiexec -n <num_processes> ./poisson_solver_rma <grid_size> [max_iterations] [exchange_method]
```

Where:
- `<num_processes>`: Number of MPI processes (should ideally be a perfect square for good domain decomposition)
- `<grid_size>`: Size of the global grid (N×N)
- `[max_iterations]`: Maximum number of Jacobi iterations (default: 10000)
- `[exchange_method]`: Method for ghost exchange:
  - 0: Traditional MPI message passing (default)
  - 1: RMA with MPI_Win_fence synchronization
  - 2: RMA with general active target synchronization

## Examples

Run with MPI message passing:
```bash
make run
```

Run with RMA fence synchronization:
```bash
make run_fence
```

Run with RMA general active target synchronization:
```bash
make run_gats
```

Run all tests with a smaller grid for quick comparison:
```bash
make test_all
```

## Understanding the Output

The program will output:
- Number of iterations required for convergence
- Final residual error
- Grid size and processor grid configuration
- Exchange method used
- Setup time, solve time, and total execution time

## Technical Details

### 2D Domain Decomposition

The global NxN grid is divided among MPI processes in a 2D Cartesian topology. Each process is responsible for computing a local portion of the grid and exchanging ghost/halo cells with its neighbors.

### Ghost/Halo Exchange Methods

1. **MPI Message Passing**:
   - Uses non-blocking MPI_Isend and MPI_Irecv operations
   - Requires explicit message passing between processes

2. **RMA with MPI_Win_fence**:
   - Uses one-sided communication with MPI_Put operations
   - Synchronization is performed using MPI_Win_fence calls
   - Simpler to implement but potentially less efficient

3. **RMA with General Active Target Synchronization**:
   - Uses MPI_Win_start, MPI_Win_complete, MPI_Win_post, and MPI_Win_wait
   - Allows more fine-grained control over synchronization
   - Potentially more efficient by reducing global synchronization

### Jacobi Iteration

The program uses Jacobi iteration to solve the discretized Poisson equation:
- Each grid point is updated based on its four neighbors and the source term
- Convergence is determined by calculating the global residual error

### Efficiency Comparison

You can compare the performance of the three ghost exchange methods by running the program with different exchange_method values. Generally, RMA operations can be more efficient than traditional message passing in certain communication patterns, but actual performance depends on hardware, MPI implementation, and problem characteristics.





Common Parameters:

Grid size: 64×64 points
Maximum iterations: 1000
Using 4 MPI processes arranged in a 2×2 processor grid
Each run shows convergence behavior with residual values printed every 100 iterations


Three Communication Methods Compared:

Standard MPI Send/Recv (method 0)
One-sided RMA (Remote Memory Access) with fence synchronization (method 1)
One-sided RMA with general active target synchronization (method 2)


Results for Each Method:

All methods achieved identical numerical results (same residual values)
The solution converged after 1000 iterations with a final residual of 2.300506e+00
Performance varied between methods:

Standard MPI: 0.0057s solve time (fastest)
RMA fence: 0.0188s solve time (slowest)
RMA general active target: 0.0174s solve time




Performance Comparison:

Traditional MPI Send/Recv was about 3.3× faster than the RMA methods
The RMA general active target was slightly faster than RMA fence



This output demonstrates that while all three communication methods produce identical numerical results, the traditional point-to-point communication (Send/Recv) outperformed the one-sided communication methods for this particular problem size and configuration. This might be unexpected since RMA is often assumed to be more efficient, but its performance depends heavily on the specific implementation, hardware, and problem characteristics.
For this small problem size (64×64 grid), the overhead of setting up RMA windows and synchronization likely outweighed any potential benefits of the one-sided communication model.


mpicc -o ax ax.c
mpirun -np 4 ./ax


Ax = [235.0 252.0 282.0 316.0 194.0 226.0 263.0 189.0 210.0 192.0 130.0 147.0 163.0 188.0 254.0 245.0 191.0 187.0 197.0 221.0 ]