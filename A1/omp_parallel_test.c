/* Example of using "parallel" block */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
	// task: add 10 to each element of x
	int N = 9;
	int x[] = {0,1,2,3,4,5,6,7,8};

	int i;
	// print initial data
	for (i = 0; i < N; i++) {
		printf("%d ", x[i]);
	}
	printf("\n");

	// process data
	#pragma omp parallel num_threads(4)
	{
		int tid = omp_get_thread_num();
		int chunk_size = N / omp_get_num_threads();
		int start_idx = tid * chunk_size;
		int end_idx = (tid == omp_get_num_threads()-1) ? N : start_idx + chunk_size;
		int idx;
		for (idx = start_idx; idx < end_idx; idx++) {
			printf("thread %d processed element %d of x\n", omp_get_thread_num(), idx);
			x[idx] += 10;
		}
	}

	// print results
    for (i = 0; i < N; i++) {
        printf("%d ", x[i]);
    }
    printf("\n");

    // done
    exit(0);
}
