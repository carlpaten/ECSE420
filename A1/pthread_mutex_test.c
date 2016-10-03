#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#define NUM_THREADS 20
#define NUM_COUNTS 1000

typedef struct {
	int *count;
	pthread_mutex_t *lock_count;
} thread_arg_t;

void *count_up(void *arg) {
	thread_arg_t *thread_arg = (thread_arg_t *) arg;
	int *count = thread_arg->count;
	pthread_mutex_t *lock_count = thread_arg->lock_count;

	// do some busy work
	int i;
	for (i = 0; i < NUM_COUNTS; i++) {
		pthread_mutex_lock(lock_count); 
		*count += 1;
		pthread_mutex_unlock(lock_count); 
	}	

	pthread_exit(NULL);
}

int main() {
	// task: increment a shared counter
	int count = 0;
	pthread_mutex_t lock_count;
	pthread_mutex_init(&lock_count, NULL);

	pthread_t *thread_ids = malloc(NUM_THREADS * sizeof(pthread_t));
	thread_arg_t *thread_args = malloc(NUM_THREADS * sizeof(thread_arg_t));

	// create threads
	int i;
	for (i = 0; i < NUM_THREADS; i++) {
		thread_args[i].count = &count;
		thread_args[i].lock_count = &lock_count;
		pthread_create(&thread_ids[i], NULL, count_up, (void *)&thread_args[i]);
	}

	// join threads
	for (i = 0; i < NUM_THREADS; i++) {
		pthread_join(thread_ids[i], NULL);
	}

	// print count-- should be NUM_THREADS*NUM_COUNTS
    printf("count: %d \n", count);

    // done
    free(thread_ids);
    free(thread_args);
    exit(0);
}
