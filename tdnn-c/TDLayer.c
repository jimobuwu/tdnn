#include "TDLayer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

TDLayer createTDLayer(unsigned int id, unsigned int neuronsCount, unsigned int delay, unsigned int inputSize) {
	TDLayer layer;
	layer.id = id;
	layer.neuronsCount = neuronsCount;
	layer.neurons = (TDNeuron*)malloc(sizeof(TDNeuron) * neuronsCount);

	if (!layer.neurons) {
		exit(1);
	}

	for (int i = 0; i < neuronsCount; ++i) {
		layer.neurons[i] = createTDNeuron((delay + 1) * inputSize);
	}

	layer.delay = delay;
	layer.inputSize = inputSize;
	layer.inputFramesSize = (layer.delay + 1) * layer.inputSize;
	layer.inputFrames = (float*)calloc(layer.inputFramesSize, sizeof(float));
	if (!layer.inputFrames) {
		exit(1);
	}
	
	return layer;
}

static void layer_pushFrame(TDLayer *layer, float *input) {
	// 从第二帧开始前移
	for (unsigned int i = 0; i < layer->delay; ++i) {
		memcpy(&layer->inputFrames[i * layer->inputSize], 
			&layer->inputFrames[(i + 1) * layer->inputSize], 
			sizeof(float) * layer->inputSize);
	}

	// 加入新帧
	memcpy(&layer->inputFrames[layer->delay * layer->inputSize], input, sizeof(float) * layer->inputSize);
}

void layer_forward(TDLayer *layer, float *input, float *output){
	layer_pushFrame(layer, input);

	if (2 > layer->id)
		printf("\nhidden layer %d: ", layer->id + 1);
	
	for (int i = 0; i < layer->neuronsCount; ++i) {
		output[i] = neuron_forward(&layer->neurons[i], layer->inputFrames);
		if (2 > layer->id)
			printf("%.4f ", output[i]);
	}
}



