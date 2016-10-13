CFLAGS += -std=c++14
CFLAGS += -fopenmp
CFLAGS += -Wall
CFLAGS += -Wextra

ass1_omp: ass1_omp.cpp lodepng.cpp
	g++ -fopenmp $(CFLAGS) lodepng.cpp ass1_omp.cpp -o ass1_omp

clean:
	rm ass1_omp
