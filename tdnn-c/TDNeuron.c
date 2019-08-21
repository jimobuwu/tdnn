#include "TDNeuron.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

TDNeuron createTDNeuron(ACTIVATION_TYPE act_type, const TDShape *kernel_shape, 
	const int *time_offsets, unsigned int offsets_size, unsigned int height_out) {
	TDNeuron neuron;
	neuron.kernel_shape = (TDShape*)malloc(sizeof(TDShape));
	memcpy(neuron.kernel_shape, kernel_shape, sizeof(TDShape));
	neuron.height_out = height_out;
	//neuron.act_type = act_type;
	neuron.act_type = NONE;
	neuron.activation = (float*)calloc(height_out, sizeof(float));
	neuron.stride_h = 1;

	if (!neuron.activation) {
		exit(1);
	}

	unsigned int kernel_flatten_size = kernel_shape->w * kernel_shape->h * kernel_shape->c;
	neuron.weights = (float*)malloc(sizeof(float) * kernel_flatten_size);
	if (!neuron.weights) {
		exit(1);
	}

	neuron.time_offsets = (int*)malloc(sizeof(int) * offsets_size);
	memcpy(neuron.time_offsets, time_offsets, sizeof(time_offsets[0]) * offsets_size);
	
	return neuron;
}

static float relu(float input) {
	return fmax(0.f, input);
}

static float sigmoid(float input) {
	return 1.f / (1.f + expf(-input));
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

// bn
static float batchNormalize(TDNeuron *neuron, float input) {

	// TODO: scale、swift
	return (input - neuron->bn_mean) / sqrt(neuron->bn_var + neuron->bn_epsilon);
}

// 根据timeoffsets下采样输入数据
static float* sampleInput(const TDNeuron *neuron, float *input, const TDShape *input_shape) {
	float *sampled = (float*)malloc(sizeof(float) * neuron->kernel_shape->w * input_shape->h);
	
	for (int i = 0; i < neuron->kernel_shape->w; ++i) {
		int offset = neuron->time_offsets[i] - neuron->time_offsets[0];
		memcpy(sampled + i * input_shape->h, input + offset * input_shape->h, sizeof(float) * input_shape->h);
	}

	return sampled;
}

void neuron_forward(TDNeuron *neuron, float *input, const TDShape *input_shape) {
	float *sampledInput = sampleInput(neuron, input, input_shape);

	float *output = getConv(sampledInput, input_shape, neuron->weights, neuron->kernel_shape, neuron->stride_h);
	for (unsigned int i = 0; i < neuron->height_out; ++i) {
		neuron->activation[i] = activate(neuron->act_type, output[i]);
	}
}

