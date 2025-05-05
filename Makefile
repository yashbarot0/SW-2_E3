CC = mpicc
CFLAGS = -O3 -Wall -lm
TARGET = poisson_solver_rma
AX_TARGET = ax

all: $(TARGET) $(AX_TARGET)

$(TARGET): poisson_solver_rma.c
	$(CC) $(CFLAGS) -o $@ $<

$(AX_TARGET): ax.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET) $(AX_TARGET)

run:
	mpiexec -n 4 ./$(TARGET) 100 10000 0

run_fence:
	mpiexec -n 4 ./$(TARGET) 100 10000 1

run_gats:
	mpiexec -n 4 ./$(TARGET) 100 10000 2

run_ax:
	mpirun -np 4 ./$(AX_TARGET)

test_all: $(TARGET)
	@echo "Running with MPI message passing..."
	mpiexec -n 4 ./$(TARGET) 64 1000 0
	@echo "\nRunning with RMA fence..."
	mpiexec -n 4 ./$(TARGET) 64 1000 1
	@echo "\nRunning with RMA general active target..."
	mpiexec -n 4 ./$(TARGET) 64 1000 2