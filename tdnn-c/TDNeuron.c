#include "TDNeuron.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

TDNeuron createTDNeuron(const TDShape *kernel_shape, const float *time_offsets, unsigned int offsets_size, unsigned int height_out) {
	TDNeuron neuron;
	neuron.kernel_shape = (TDShape*)malloc(sizeof(TDShape));
	memcpy(neuron.kernel_shape, kernel_shape, sizeof(TDShape));
	neuron.height_out = height_out;
	neuron.activation = (float*)calloc(height_out, sizeof(float));
	if (!neuron.activation) {
		exit(1);
	}

	unsigned int kernel_flatten_size = kernel_shape->w * kernel_shape->h * kernel_shape->c;
	neuron.weights = (float*)malloc(sizeof(float) * kernel_flatten_size);
	if (!neuron.weights) {
		exit(1);
	}

	for (int i = 0; i < kernel_flatten_size; ++i) {
		neuron.weights[i] = rand() / (float)RAND_MAX; // 0~1Ëæ»úÊý
	}

	memcpy(neuron.time_offsets, time_offsets, sizeof(float) * offsets_size);
	
	return neuron;
}

static float relu(float input) {
	return input;
}

static float sigmoid(float input) {
	return 1.f / (1.f + expf(-input));
}

static float activate(ACTIVATION_TYPE type, float input) {
	float output = 0.f;

	switch (type) {
	case RELU:
		output = relu(input);
		break;
	case SIGMOID:
		output = sigmoid(input);
		break;
	}

	return output;
}

void neuron_forward(TDNeuron *neuron, float *input, const TDShape *input_shape) {
	float *output = getConv(input, input_shape, neuron->weights, neuron->kernel_shape, neuron->stride_h);
	for (unsigned int i = 0; i < neuron->height_out; ++i) {
		neuron->activation[i] = activate(neuron->act_type, output[i]);
	}
}

