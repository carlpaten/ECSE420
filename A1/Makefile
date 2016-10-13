CFLAGS += -std=c++14
CFLAGS += -fopenmp
CFLAGS += -Wall
CFLAGS += -Wextra

.PHONY: all
all: ass1_omp
	chmod a+x convolve rectify pool

ass1_omp: ass1_omp.cpp lodepng.cpp
	g++ -fopenmp $(CFLAGS) lodepng.cpp ass1_omp.cpp -o ass1_omp

clean:
	rm ass1_omp
