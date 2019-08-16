#include <stdio.h>
#include "TDNet.h"
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>
#include <omp.h>


// hounet
static TDNet createHouNet() {
	TDNet net = createTDNet(1e-07, 1);

	// conv1
	TDShape kernel_shape1 = { 8, 18, 3 };
	TDShape input_shape1 = {1, 26, 3};
	float time_offsets1[8] = { -4, -3, -2, -1, 0, 1, 2, 3 };
	TDLayer l = createTDLayer(0, CONV, RELU, 128, &kernel_shape1, time_offsets1, 8, &input_shape1, 9);
	addTDLayer(&net, &l);

	//conv2
	TDShape kernel_shape2 = { 4, 3, 128 };
	TDShape input_shape2 = { 1, 9, 128 };
	float time_offsets2[4] = { -2, -1, 0, 1};
	l = createTDLayer(1, CONV, RELU, 64, &kernel_shape2, time_offsets2, 4, &input_shape2, 7);
	addTDLayer(&net, &l);

	//conv3
	TDShape kernel_shape3 = { 4, 3, 64 };
	TDShape input_shape3 = { 1, 7, 64 };
	float time_offsets3[4] = { -2, -1, 0, 1 };
	l = createTDLayer(2, CONV, RELU, 64, &kernel_shape3, time_offsets3, 4, &input_shape3, 5);
	addTDLayer(&net, &l);

	// Affine1
	/*TDShape kernel_shape4 = { 4, 3, 64 };
	TDShape input_shape4 = { 1, 9, 128 };
	float time_offsets4[4] = { -2, -1, 0, 1 };
	l = createTDLayer(2, CONV, 64, &kernel_shape3, time_offsets3, 4, &input_shape3);
	addTDLayer(&net, &l);*/

	// Affine2
	// Affine3
	// final Affine
	// softmax

	return net;
}

static void randomInput(float *input) {
	printf("input: ");
	for (int i = 0; i < 16; ++i) {
		input[i] = rand() / (float)RAND_MAX;
		printf("%f ", input[i]);
	}
}

long long getMSec() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

int main() {

// -fopenmp
#ifndef _OPENMP
	printf("not support openmp");
#else
	printf("support openmp");
#endif

	TDNet net = createHouNet();
	float input[16] = { 0.f };

	for (int i = 0; i < 16; ++i) {
		float *output = forward(&net, input);
		printf("\noutput: ");

		for (int j = 0; j < 3; ++j) {
			printf("%.4f ", output[j]);			
		}
	}

	return 0;
}

// test openmp
//int main() {
//	long long begin, end = 0;
//	begin = getMSec();
//#pragma omp parallel for
//	for (int i = 0; i < 10000000; ++i) {
//		int a = 1;
//	}
//	end = getMSec();
//	printf("\ncost time: %lld", end - begin);
//}