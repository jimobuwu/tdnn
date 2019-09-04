#include "TDNeuron.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "TDActivation.h"

TDNeuron createTDNeuron(ACTIVATION_TYPE actType, const TDShape *kernelShape, 
	const int *timeOffsets, unsigned int offsetsSize, unsigned int heightOut,
	_Bool actBeforeNorm) {
	TDNeuron neuron;
	neuron.kernelShape = (TDShape*)malloc(sizeof(TDShape));
	if (!neuron.kernelShape) {
		exit(1);
	}

	memcpy(neuron.kernelShape, kernelShape, sizeof(TDShape));
	neuron.heightOut = heightOut;
	neuron.actType = actType;
	neuron.activation = (float*)calloc(heightOut, sizeof(float));
	neuron.stride_h = 1;
	neuron.bn = NULL;
	neuron.norm_type = NONE_ACT;
	neuron.actBeforeNorm = actBeforeNorm;

	if (!neuron.activation) {
		exit(1);
	}

	unsigned int kernel_flatten_size = kernelShape->w * kernelShape->h * kernelShape->c;
	neuron.weights = (float*)malloc(sizeof(float) * kernel_flatten_size);
	if (!neuron.weights) {
		exit(1);
	}

	neuron.timeOffsets = (int*)malloc(sizeof(int) * offsetsSize);
	if (!neuron.timeOffsets) {
		exit(1);
	}

	memcpy(neuron.timeOffsets, timeOffsets, sizeof(timeOffsets[0]) * offsetsSize);
	
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

void addNeuronBN(TDNeuron * neuron, float epsilon, float gamma, float mean, float var) {
	neuron->norm_type = BN;
	neuron->bn = (TDNeuronBatchNorm*)malloc(sizeof(TDNeuronBatchNorm));
	neuron->bn->epsilon = epsilon;
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

static float normalize(TDNeuron *neuron, float input) {
	float output = input;

	switch (neuron->norm_type) {
	case BN:
		output = batchNormalize(neuron, input);
		break;	
	default:
		break;
	}

	return output;
}

void neuron_forward(TDNeuron *neuron, float *input, const TDShape *inputShape) {
	float *output = getConv(input, inputShape, neuron->weights, neuron->kernelShape, neuron->stride_h);
	for (unsigned int i = 0; i < neuron->heightOut; ++i) {
		if (neuron->actBeforeNorm)
			neuron->activation[i] = normalize(neuron, activate(neuron->actType, output[i] + neuron->bias));
		else
			neuron->activation[i] = activate(neuron->actType, normalize(neuron, output[i] + neuron->bias));
	}
}

