#include "TDLayer.h"

TDLayer createTDLayer(int neuronsCount, int delay, int inputSize) {
	TDLayer layer;
	layer.neuronsCount = neuronsCount;
	layer.neurons = (TDNeuron*)malloc(sizeof(TDNeuron) * neuronsCount);
	for (int i = 0; i < neuronsCount; ++i) {
		layer.neurons[i] = createTDNeuron((delay + 1) * inputSize);
	}

	return layer;
}

