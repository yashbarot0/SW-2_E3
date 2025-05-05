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