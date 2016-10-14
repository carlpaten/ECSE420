CFLAGS += -std=c++14
CFLAGS += -fopenmp
CFLAGS += -Wall
CFLAGS += -Wextra
CFLAGS += -O3

.PHONY: all
all: ass1_omp
	chmod a+x convolve rectify pool

ass1_omp: ass1_omp.cpp lodepng.cpp
	g++ -fopenmp $(CFLAGS) lodepng.cpp ass1_omp.cpp -o ass1_omp

clean:
	rm ass1_omp

.PHONY: test_convolve
test_convolve: ass1_omp
	./convolve test.png test_convolve2.png
	./test_equality test_convolve.png test_convolve2.png
