#include "TDLayer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include "TDUtils.h"

TDLayer createTDLayer(unsigned int id, const char* name, LAYER_TYPE layer_type, ACTIVATION_TYPE act_type, unsigned int neuronsCount, const TDShape *kernel_shape,
	const int *time_offsets, unsigned int offsets_size, const TDShape *input_shape, unsigned int height_out) {
	TDLayer layer;
	layer.id = id;
	layer.name = malloc(strlen(name) + 1);
	strcpy(layer.name, name);
	layer.type = layer_type;
	layer.neuronsCount = neuronsCount;
	layer.delay = kernel_shape->w;

	layer.input_shape = (TDShape*)malloc(sizeof(TDShape));
	if (!layer.input_shape) {
		exit(1);
	}
	memcpy(layer.input_shape, input_shape, sizeof(TDShape));

	layer.neurons = (TDNeuron*)malloc(sizeof(TDNeuron) * neuronsCount);
	if (!layer.neurons) {
		exit(1);
	}

//#pragma omp parallel for
	for (int i = 0; i < neuronsCount; ++i) {
		layer.neurons[i] = createTDNeuron(act_type, kernel_shape, time_offsets, offsets_size, height_out);
	}

	unsigned int inputFramesSize = (time_offsets[offsets_size - 1] - time_offsets[0] + 1) * layer.input_shape->c * layer.input_shape->h;
	layer.inputFramesSize = inputFramesSize;
	layer.inputFrames = (float*)calloc(inputFramesSize, sizeof(float));
	if (!layer.inputFrames) {
		exit(1);
	}

	return layer;
}

static void layer_pushFrame(TDLayer *layer, float *input) {
	// 从第二帧开始前移
	unsigned int input_size = layer->input_shape->h * layer->input_shape->c;
	for (unsigned int i = 0; i < layer->delay - 1; ++i) {
		memcpy(&layer->inputFrames[i * input_size],
			&layer->inputFrames[(i + 1) * input_size],
			sizeof(float) * input_size);
	}

	// 加入新帧
	memcpy(&layer->inputFrames[(layer->delay - 1) * input_size], input, sizeof(float) * input_size);
}

void layer_forward(TDLayer *layer, const float *input, float *output) {
	layer_pushFrame(layer, input);

	unsigned int height_out = layer->neurons[0].height_out;
	printf("\n\n input shape: { %d, %d, %d } \n", 1, layer->input_shape->h, layer->input_shape->c);
	printf("output shape: { %d, %d, %d }", 1, height_out, layer->neuronsCount);
	
	for (unsigned int i = 0; i < layer->neuronsCount; ++i) {
		neuron_forward(&layer->neurons[i], layer->inputFrames, layer->input_shape);

		for (unsigned int j = 0; j < height_out; ++j) {
			output[i * height_out + j] = layer->neurons[i].activation[j];
		}
	}
}

void load_weights(TDLayer * layer, const char *filePath) {
	TDShape * shape = layer->neurons[0].kernel_shape;
	unsigned flatten = shape->w * shape->h * shape->c;
	unsigned total = flatten * layer->neuronsCount;

	float *linear_weights = (float*)malloc(sizeof(float) * total);
	float *bias_weights = (float*)malloc(sizeof(float) * layer->neuronsCount);

	/*char *fileName = layer->name;
	size_t len = strlen(fileName);
	fileName = (char*)realloc(fileName, strlen(fileName) + 5);
	strcat(fileName, ".txt");*/

	parseWeights(filePath, layer->neuronsCount, linear_weights, bias_weights);

	unsigned index = 0;
	for (int i = 0; i < layer->neuronsCount; ++i, index = i * flatten) {
		memcpy(layer->neurons[i].weights, linear_weights + index, sizeof(float) * flatten);
		layer->neurons[i].bias = bias_weights[i];
	}
}



