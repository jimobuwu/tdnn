#include "TDNeuron.h"
#include <stdlib.h>

TDNeuron createTDNeuron(int nConnections) {
	TDNeuron neuron;
	neuron.nConnections = nConnections;
	neuron.weights = (float*)malloc(sizeof(float*) * nConnections);
	
	for (int i = 0; i < nConnections; ++i) {
		neuron.weights[i] = rand() / ((RAND_MAX + 1.0)); // 0~1Ëæ»úÊý
	}
}

float sigmoid(float input) {
	return 1 / (1 + exp(-input));
}

float sigmoidDerivative(float input) {
	return sigmoid(input) * ( 1 - sigmoid(input));
}

float activate(float input) {
	return sigmoid(input);
}

float derivative(float input) {
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

