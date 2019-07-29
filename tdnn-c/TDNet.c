#include "TDNet.h"


TDNet createTDNet() {

	TDNet net;
	net.layersCount = 3;
	net.learningRate = 0.01;
	net.decayRate = 0.01;
	net.layers = (TDLayer*)malloc(sizeof(TDLayer*) * net.layersCount);

	TDLayer layer;
	layer.neuronsCount = 8;
	layer.neurons = (TDNeuron*)malloc(sizeof(TDNeuron*) * 8);
	net.layers[0] = layer;

	return net;
}
