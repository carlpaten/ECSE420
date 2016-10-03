/* Example of using lodepng to load, process, save image */
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>

#include <time.h>

enum multi_lib { PTHREADS, OPENMP };

void rectify(int count, unsigned char* image, int threads, enum multi_lib lib) {
	printf("Pixel count is %d\n", count);
	clock_t t1 = clock();
	#pragma omp parallel for num_threads(threads)
	for(int i = 0; i < count; i++) {
		//printf("Processing %d\n", i);
		image[i] = image[i] > 127 ? image[i] : 127;
	}
	clock_t t2 = clock();
	printf("The time taken is %ld\n", (t2 - t1));
}

void process(char* input_filename, char* output_filename, int threads, enum multi_lib lib)
{
  unsigned error;
  unsigned char *image;
  unsigned width, height;

  error = lodepng_decode32_file(&image, &width, &height, input_filename);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
  rectify(width * height * 4, image, threads, lib);
  lodepng_encode32_file(output_filename, image, width, height);

  free(image);
}

int main(int argc, char *argv[])
{
  char* input_filename = argv[1];
  char* output_filename = argv[2];
  int threads = atoi(argv[3]);
  enum multi_lib lib = OPENMP;
  if(argc >= 5) {
	char* lib = argv[4];
	if(strcmp(lib, "pthreads") == 0) {
		printf("Using pthreads");
		lib = PTHREADS;
	}
  }

  process(input_filename, output_filename, threads, lib);

  return 0;
}
