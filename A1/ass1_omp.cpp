#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

#include <omp.h>
#include <time.h>

#include "lodepng.h"
#include "wm.h"

struct Pixel {
	unsigned char R;
	unsigned char G;
	unsigned char B;
	unsigned char A;
};

class Image {
	public:
		unsigned int width;
		unsigned int height;
		Pixel* raw;
		Image(unsigned int width, unsigned int height);
		Image(const std::string filename);
		Pixel* get(unsigned int x, unsigned int y);
		void save(const std::string filename);
};

Image::Image(unsigned int _width, unsigned int _height) : 
	width(_width), 
	height(_height),
	raw(new Pixel[width * height])
{
};

Image::Image(std::string filename) {
	unsigned char* raw_as_chars;
	auto error = lodepng_decode32_file(&raw_as_chars, &width, &height, filename.c_str());
	if(error) throw new std::runtime_error(lodepng_error_text(error));
	raw = new Pixel[width * height];
	memcpy(raw, raw_as_chars, width * height * sizeof(Pixel));
}

Pixel* Image::get(unsigned int x, unsigned int y) {
	return &(raw[x + y * width]);
}

void Image::save(const std::string filename) {
	unsigned char* raw_as_chars = new unsigned char[width * height * sizeof(Pixel)];
	memcpy(raw_as_chars, raw, width * height * sizeof(Pixel));
	auto error = lodepng_encode32_file(filename.c_str(), raw_as_chars, width, height);
	if(error) throw new std::runtime_error(lodepng_error_text(error));
}

template<typename T>
T clamp(T val, T min, T max) {
	if(val < min) {
		return min;
	} else if(val > max) {
		return max;
	} else {
		return val;
	}
}

void rectify(Image* input, Image* output, unsigned char ceiling, unsigned int threads = 1) {
	#pragma omp parallel for num_threads(threads)
	for(unsigned int i = 0; i < output->width; i++) {
		for(unsigned int j = 0; j < output->height; j++) {

			//Pixel* p = &(img->raw[i]);
			Pixel* pi = input->get(i, j);
			Pixel* po = output->get(i, j);
			po->R = clamp<unsigned char>(pi->R, ceiling, 255);
			po->G = clamp<unsigned char>(pi->G, ceiling, 255);
			po->B = clamp<unsigned char>(pi->B, ceiling, 255);
			po->A = 255;
		}
	}
}

void pool(Image* input, Image* output, unsigned int threads = 1) {
	#pragma omp parallel for num_threads(threads)
	for(unsigned int i = 0; i < output->width; i++) {
		for(unsigned int j = 0; j < output->height; j++) {
			Pixel* p1 = input->get(2 * i    , 2 * j    );
			Pixel* p2 = input->get(2 * i + 1, 2 * j    );
			Pixel* p3 = input->get(2 * i    , 2 * j + 1);
			Pixel* p4 = input->get(2 * i + 1, 2 * j + 1);
			Pixel* po = output->get(i, j);
			po->R = p1->R < p2->R ? p2->R < p3->R ? p3->R < p4->R ? p4->R : p3->R : p2->R : p1->R;
			po->G = p1->G < p2->G ? p2->G < p3->G ? p3->G < p4->G ? p4->G : p3->G : p2->G : p1->G;
			po->B = p1->B < p2->B ? p2->B < p3->B ? p3->B < p4->B ? p4->B : p3->B : p2->B : p1->B;
			po->A = 255;
		}
	}
}

void convolve(Image* input, Image* output, float weightmatrix[3][3], unsigned int threads = 1) {
	auto wm = weightmatrix; //shorthand

	#pragma omp parallel for num_threads(threads)
	for(unsigned int i = 0; i < output->width; i++) {
		for(unsigned int j = 0; j < output->height; j++) {
			float rSum = 0;
			float gSum = 0;
			float bSum = 0;
			for(int k = 0; k <= 2; k++) {
				for(int l = 0; l <= 2; l++) {
					Pixel* p = input->get(i + k, j + l);
					//printf("Red contribution of (%d, %d) to (%d, %d) is %f\n", i + k - 1, j + l - 1, i, j, p->R * w[k][l]); 

					rSum += p->R * wm[l][k];
					gSum += p->G * wm[l][k];
					bSum += p->B * wm[l][k];
				}
			}
			Pixel* p = output->get(i, j);
			p->R = std::lround(clamp<float>(rSum, 0.0, 255.0));
			p->G = std::lround(clamp<float>(gSum, 0.0, 255.0));
			p->B = std::lround(clamp<float>(bSum, 0.0, 255.0));
			p->A = 255;
		}
	}
}

auto time_execution(std::function<void()> f) {
	auto t1 = clock();
	f();
	auto t2 = clock();
	return t2 - t1;
}

void symmetric_difference(Image* input1, Image* input2, Image* output) {
	for(unsigned int i = 0; i < output->width; i++) {
		for(unsigned int j = 0; j < output->height; j++) {
			Pixel* p1 = input1->get(i, j);
			Pixel* p2 = input2->get(i, j);
			Pixel* po = output->get(i, j);

			po->R = abs((int) (p1->R) - (int) (p2 -> R)); 
			po->G = abs((int) (p1->G) - (int) (p2 -> G)); 
			po->B = abs((int) (p1->B) - (int) (p2 -> B)); 
			po->A = 255;
		}
	}
}

int main(int argc, char *argv[]) {
	if(argc < 4) {
		printf("Usage: %s OPERATION INPUT_FILE OUTPUT_FILE [THREADCOUNT]\n", argv[0]);
		exit(1);
	}

	char* operation       = argv[1];
	char* input_filename  = argv[2];
	char* output_filename = argv[3];
	int threads = 1;
	if(argc >= 4) threads = atoi(argv[3]);

	printf("Loading file %s.\n", input_filename);
	Image* input = new Image(input_filename);
	Image* output;
	long time_taken = 0;

	if(!strcmp(operation, "rectification")) {
		output = new Image(input->width, input->height);
		time_taken = time_execution([input, output, threads]() { rectify(input, output, 127, threads); });
	} else if(!strcmp(operation, "max-pooling")) {
		output = new Image(input->width / 2, input->height / 2);
		time_taken = time_execution([input, output, threads]() { pool(input, output, threads); });
	} else if(!strcmp(operation, "convolution")) {
		output = new Image(input->width - 2, input->height - 2);
		time_taken = time_execution([input, output, threads]() { convolve(input, output, w, threads); });
	} else {
		printf("Unrecognized operation %s.\n", operation);
		exit(1);
	}

	printf("%s completed in %ld microseconds.\n", operation, time_taken);
	printf("Saving to file %s.\n", output_filename);
	output->save(output_filename);

	return 0;
}
