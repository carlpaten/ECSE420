#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#define NUM_THREADS 4

typedef struct {
	int tid;
	int *data;
	int data_length;
} thread_arg_t;

void *add10(void *arg) {
	thread_arg_t *thread_arg = (thread_arg_t *) arg;
	int tid = thread_arg->tid;
	int *x = thread_arg->data;
	int N = thread_arg->data_length;

	// TODO
	int chunk_size = N / NUM_THREADS;
	int start_idx = tid * chunk_size;
	int end_idx = (tid + 1) * chunk_size;
	int idx;

	for (idx = start_idx; idx < end_idx; idx++) {
		x[idx] += 10;
	}		

	pthread_exit(NULL);
	//return 0;
}

int main() {
	// task: add 10 to each element of x
	int N = 8;
	int x[] = {0,1,2,3,4,5,6,7};

	pthread_t *thread_ids = malloc(NUM_THREADS * sizeof(pthread_t));
	thread_arg_t *thread_args = malloc(NUM_THREADS * sizeof(thread_arg_t));

	int i;
	// print initial data
	for (i = 0; i < N; i++) {
		printf("%d ", x[i]);
	}
	printf("\n");

	// create threads
	for (i = 0; i < NUM_THREADS; i++) {
		thread_args[i].tid = i;
		thread_args[i].data = x;
		thread_args[i].data_length = N;
		pthread_create(&thread_ids[i], NULL, add10, (void *)&thread_args[i]);
	}

	// join threads
	for (i = 0; i < NUM_THREADS; i++) {
		pthread_join(thread_ids[i], NULL);
	}

	// print results
    for (i = 0; i < N; i++) {
        printf("%d ", x[i]);
    }
    printf("\n");

    // done
    free(thread_ids);
    free(thread_args);
    exit(0);
}
