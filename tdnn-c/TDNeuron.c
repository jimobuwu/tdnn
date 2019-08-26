#include "TDNeuron.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "TDActivation.h"

TDNeuron createTDNeuron(ACTIVATION_TYPE act_type, const TDShape *kernel_shape, 
	const int *time_offsets, unsigned int offsets_size, unsigned int height_out) {
	TDNeuron neuron;
	neuron.kernel_shape = (TDShape*)malloc(sizeof(TDShape));
	if (!neuron.kernel_shape) {
		exit(1);
	}

	memcpy(neuron.kernel_shape, kernel_shape, sizeof(TDShape));
	neuron.height_out = height_out;
	neuron.act_type = act_type;
	neuron.activation = (float*)calloc(height_out, sizeof(float));
	neuron.stride_h = 1;
	neuron.bn = NULL;

	if (!neuron.activation) {
		exit(1);
	}

	unsigned int kernel_flatten_size = kernel_shape->w * kernel_shape->h * kernel_shape->c;
	neuron.weights = (float*)malloc(sizeof(float) * kernel_flatten_size);
	if (!neuron.weights) {
		exit(1);
	}

	neuron.time_offsets = (int*)malloc(sizeof(int) * offsets_size);
	if (!neuron.time_offsets) {
		exit(1);
	}

	memcpy(neuron.time_offsets, time_offsets, sizeof(time_offsets[0]) * offsets_size);
	
	return neuron;
}

static float activate(ACTIVATION_TYPE type, float input) {
	float output = input;

	switch (type) {
	case RELU:
		output = relu(input);
		break;
	case SIGMOID:
		output = sigmoid(input);
		break;
	default:
		break;
	}

	return output;
}

void addNeuronBN(TDNeuron * neuron, float epsilon, unsigned count, float gamma, float mean, float var) {
	neuron->bn = (TDNeuronBatchNorm*)malloc(sizeof(TDNeuronBatchNorm));
	neuron->bn->epsilon = epsilon;
	neuron->bn->count = count;
	neuron->bn->gamma = gamma;
	neuron->bn->mean = mean;
	neuron->bn->var = var;
	
	neuron->bn->scale = gamma * powf(fmax(0.f, var) + epsilon, -0.5);
	neuron->bn->offset = -neuron->bn->scale * mean;
}

// bn
static float batchNormalize(TDNeuron *neuron, float input) {
	return neuron->bn->scale * input + neuron->bn->offset;
}

void neuron_forward(TDNeuron *neuron, float *input, const TDShape *input_shape) {
	float *output = getConv(input, input_shape, neuron->weights, neuron->kernel_shape, neuron->stride_h);
	for (unsigned int i = 0; i < neuron->height_out; ++i) {
		if(neuron->bn)
			neuron->activation[i] = activate(neuron->act_type, batchNormalize(neuron, output[i] + neuron->bias));
		else
			neuron->activation[i] = activate(neuron->act_type, output[i] + neuron->bias);
	}
}

