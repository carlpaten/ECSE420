CFLAGS += -std=gnu11
CFLAGS += -fopenmp
CFLAGS += -pthread
CFLAGS += -Wall
CFLAGS += -Wextra

all: rectify_omp rectify

rectify: rectify_omp
	cp rectify_omp rectify

rectify_omp: rectify_omp.c
	gcc $(CFLAGS) lodepng.c rectify_omp.c -o rectify_omp

clean:
	rm rectify rectify_omp
