#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define NUM_THREADS 20
#define NUM_COUNTS 1000

int main() {
	// task: increment a shared counter
	int count = 0;

	#pragma omp parallel num_threads(NUM_THREADS)
	{
		// do some busy work
		int i;
		for (i = 0; i < NUM_COUNTS; i++) {
			#pragma omp atomic 
			count += 1;
		}
	}

	// print count-- should be NUM_THREADS*NUM_COUNTS
    printf("count: %d \n",count);

    // done
    exit(0);
}
