#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
	// task: find maximum element of x
	int N = 8;
	int x[] = {0,1,2,3,4,5,6,7};
	int max_val = 0;

	int i;
	// print initial data
	for (i = 0; i < N; i++) {
		printf("%d ", x[i]);
	}
	printf("\n");

	// process data
	int idx;
	#pragma omp parallel for reduction(max : max_val)
	for (idx = 0; idx < N; idx++) {
		if (x[idx] > max_val) {
			max_val = x[idx];
		}
	}

	// print result
    printf("max: %d \n", max_val);

    // done
    exit(0);
}
