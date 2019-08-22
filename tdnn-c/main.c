#include <stdio.h>
#include "TDNet.h"
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>
#include <omp.h>

// hounet
static TDNet createHouNet() {
	TDNet net = createTDNet(1);

	net.input_dim = 78;

	// conv1
	TDShape kernel_shape1 = { 8, 18, 3 };
	TDShape input_shape1 = {1, 26, 3};
	int time_offsets1[8] = { -4, -3, -2, -1, 0, 1, 2, 3 };
	TDLayer l1 = createTDLayer(0, "cnn1.conv", CONV, NONE_ACT, 128, &kernel_shape1, time_offsets1, 8, &input_shape1, 9, 0);
	load_weights(&l1, "../../../data/hounet/cnn1.conv.txt");
	addBN(&l1, "../../../data/hounet/cnn1.batchnorm.txt", 128, 0.001, 1008000, 1);
	addTDLayer(&net, &l1);

	////conv2
	//TDShape kernel_shape2 = { 4, 3, 128 };
	//TDShape input_shape2 = { 1, 9, 128 };
	//int time_offsets2[4] = { -2, -1, 0, 1};
	//TDLayer l2 = createTDLayer(1, "cnn2.conv", CONV, RELU, 64, &kernel_shape2, time_offsets2, 4, &input_shape2, 7, 0);
	//load_weights(&l2, "../../../data/hounet/cnn2.conv.txt");
	//addBN(&l2, "../../../data/hounet/cnn1.batchnorm.txt", 64, 0.001, 700000.1, 1);
	//addTDLayer(&net, &l2);

	//////conv3
	//TDShape kernel_shape3 = { 4, 3, 64 };
	//TDShape input_shape3 = { 1, 7, 64 };
	//int time_offsets3[4] = { -2, -1, 0, 1 };
	//TDLayer l3 = createTDLayer(2, "cnn3.conv", CONV, RELU, 64, &kernel_shape3, time_offsets3, 4, &input_shape3, 5, 0);
	//load_weights(&l3, "../../../data/hounet/cnn3.conv.txt");
	//addBN(&l3, "../../../data/hounet/cnn1.batchnorm.txt", 64, 0.001, 440000, 1);
	//addTDLayer(&net, &l3);
	//
	//// Affine1
	//TDShape kernel_shape4 = { 3, 5, 64 };
	//TDShape input_shape4 = { 1, 5, 64 };
	//int time_offsets4[4] = { -1, 0, 1 };
	//TDLayer l4 = createTDLayer(3, "Affine1", DENSE, RELU, 512, &kernel_shape4, time_offsets4, 3, &input_shape4, 1, 0);
	//load_weights(&l4, "../../../data/hounet/Affine1.txt");
	//addTDLayer(&net, &l4);

	//// Affine2
	//TDShape kernel_shape5 = { 1, 512, 1 };
	//TDShape input_shape5 = { 1, 512, 1 };
	//int time_offsets5[1] = { 0 };
	//TDLayer l5 = createTDLayer(4, "Affine2", DENSE, RELU, 512, &kernel_shape5, time_offsets5, 1, &input_shape5, 1, 0);
	//load_weights(&l5, "../../../data/hounet/Affine2.txt");
	//addTDLayer(&net, &l5);
	//// Affine3
	//TDShape kernel_shape6 = { 1, 512, 1 };
	//TDShape input_shape6 = { 1, 512, 1 };
	//int time_offsets6[1] = { 0 };
	//TDLayer l6 = createTDLayer(5, "Affine3", DENSE, RELU, 512, &kernel_shape6, time_offsets6, 1, &input_shape6, 1, 0);
	//load_weights(&l6, "../../../data/hounet/Affine3.txt");
	//addTDLayer(&net, &l6);

	//// final Affine
	///*TDShape kernel_shape6 = { 1, 512, 1 };
	//TDShape input_shape6 = { 1, 512, 1 };
	//int time_offsets6[1] = { 0 };*/
	//TDLayer l7 = createTDLayer(6, "Final_affine", DENSE, NONE_ACT, 67, &kernel_shape6, time_offsets6, 1, &input_shape6, 1, 1);
	//load_weights(&l7, "../../../data/hounet/Final_affine.txt");
	//addTDLayer(&net, &l7);

	// softmax

	return net;
}

//static void randomInput(float *input) {
//	printf("input: ");
//	for (int i = 0; i < 16; ++i) {
//		input[i] = rand() / (float)RAND_MAX;
//		printf("%f ", input[i]);
//	}
//}

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
	parseInputFile("../../../data/hounet/input.txt", &net);

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