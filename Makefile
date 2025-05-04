CC = mpicc
CFLAGS = -O3 -Wall

all: poisson_compare_fence poisson_compare_active

poisson_compare_fence: poisson_compare_fence.c
	$(CC) $(CFLAGS) -o $@ $< -lm

poisson_compare_active: poisson_compare_active.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f poisson_compare_fence poisson_compare_active
