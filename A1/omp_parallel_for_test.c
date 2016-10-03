/* Example of work sharing using "parallel for" */
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
	int idx;
	#pragma omp parallel for num_threads(4)
	for (idx = 0; idx < N; idx++) {
		printf("thread %d processed element %d of x\n", omp_get_thread_num(), idx);
		x[idx] += 10;
	}

	// print results
    for (i = 0; i < N; i++) {
        printf("%d ", x[i]);
    }
    printf("\n");

    // done
    exit(0);
}
