#include "TDLayer.h"

TDLayer createTDLayer(int neuronsCount, int delay, int inputSize) {
	TDLayer layer;
	layer.neuronsCount = neuronsCount;
	layer.neurons = (TDNeuron*)malloc(sizeof(TDNeuron) * neuronsCount);
	for (int i = 0; i < neuronsCount; ++i) {
		layer.neurons[i] = createTDNeuron((delay + 1) * inputSize);
	}

	layer.inputFramesSize = (layer.delay + 1) * layer.inputSize;
	layer.inputFrames = (float*)calloc(layer.inputFramesSize, sizeof(float));
	layer.inputSize = inputSize;

	return layer;
}

static void layer_pushFrame(TDLayer *layer, float *input) {
	// 从第二帧开始前移
	for (int i = 0; i < layer->delay; ++i) {
		memcpy(&layer->inputFrames[(i + 1) * layer->inputSize], 
			&layer->inputFrames[i * layer->inputSize], 
			layer->inputSize);
	}

	// 加入新帧
	memcpy(&layer->inputFrames[layer->delay * layer->inputSize], input, layer->inputSize);
}

void layer_forward(TDLayer *layer, float *input, float *output){
	layer_pushFrame(layer, input);

	for (int i = 0; i < layer->neuronsCount; ++i) {
		output[i] = neuron_forward(&layer->neurons[i], layer->inputFrames);
	}
}



