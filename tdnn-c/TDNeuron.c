#include "TDNeuron.h"
#include <stdlib.h>

TDNeuron createTDNeuron(int nConnections) {
	TDNeuron neuron;
	neuron.nConnections = nConnections;
	neuron.weights = (float*)malloc(sizeof(float) * nConnections);
	
	for (int i = 0; i < nConnections; ++i) {
		neuron.weights[i] = rand() / ((RAND_MAX + 1.0)); // 0~1Ëæ»úÊý
	}
}

static float sigmoid(float input) {
	return 1 / (1 + exp(-input));
}

static float sigmoidDerivative(float input) {
	return sigmoid(input) * ( 1 - sigmoid(input));
}

static float activate(float input) {
	return sigmoid(input);
}

static float derivative(float input) {
	return sigmoidDerivative(input);
}

float forward(TDNeuron* neuron, float* input) {
	float sum = 0;
	for (int i = 0; i < neuron->nConnections; ++i) {
		sum += neuron->weights[i] * input[i];
	}
	
	neuron->activation = activate(sum);
	neuron->derivate = derivative(sum);

	return neuron->activation;
}

