#include "TDLayer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include "TDUtils.h"

TDLayer createTDLayer(unsigned int id, LAYER_TYPE layer_type, unsigned int neuronsCount, const TDShape *kernel_shape,
	const float *time_offsets, unsigned int offsets_size, const TDShape *input_shape, unsigned int height_out) {
	TDLayer layer;
	layer.id = id;
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
		layer.neurons[i] = createTDNeuron(kernel_shape, time_offsets, offsets_size, height_out);
	}

	unsigned int inputFramesSize = kernel_shape->w * input_shape->h;
	layer.inputFrames = (float*)calloc(layer.inputFramesSize, sizeof(float));
	if (!layer.inputFrames) {
		exit(1);
	}
	
	return layer;
}

static void layer_pushFrame(TDLayer *layer, float *input) {
	// 从第二帧开始前移
	unsigned int input_size = layer->input_shape->h * layer->input_shape->c;
	for (unsigned int i = 0; i < layer->delay; ++i) {
		memcpy(&layer->inputFrames[i * input_size],
			&layer->inputFrames[(i + 1) * input_size],
			sizeof(float) * input_size);
	}

	// 加入新帧
	memcpy(&layer->inputFrames[layer->delay * input_size], input, sizeof(float) * input_size);
}

void layer_forward(TDLayer *layer, float *input, float *output){
	layer_pushFrame(layer, input);

	unsigned int height_out = 0;
	for (unsigned int i = 0; i < layer->neuronsCount; ++i) {
		height_out = layer->neurons[i].height_out;
		neuron_forward(&layer->neurons[i], input, layer->input_shape);

		for (unsigned int j = 0; j < height_out; ++j) {
			output[i * height_out + j] = layer->neurons[i].activation[j];
		}
	}
}



