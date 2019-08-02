#include <stdio.h>
#include "TDNet.h"
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>

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
	TDNet net = createTDNet();
	long long begin, end = 0;

	//float input[15][16] = {
	//	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
	//	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
	//	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
	//	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
	//	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
	//	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
	//	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
	//	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
	//	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
	//	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
	//	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
	//	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
	//	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
	//	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16},
	//	{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}
	//};


	const int frames = 15;
	float input[16] = { 0.f };
	srand(time(NULL));

	begin = getMSec();
	for (int i = 0; i < frames; ++i) {
		printf("\nframe: %d \n", i + 1);		
		randomInput(input);

		float* output = forward(&net, input);
		printf("\noutput: ");

		for (int j = 0; j < 3; ++j) {
			printf("%.4f ", output[j]);			
		}
		printf("\n");
	}
	end = getMSec();
	printf("\ncost time: %d", end - begin);

	return 0;
}