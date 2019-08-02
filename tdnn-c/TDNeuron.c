#include "TDNeuron.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

TDNeuron createTDNeuron(unsigned int nConnections) {
	TDNeuron neuron;
	neuron.nConnections = nConnections;
	neuron.weights = (float*)malloc(sizeof(float) * nConnections);
	neuron.activation = 0.f;
	neuron.derivate = 0.f;

	if (!neuron.weights) {
		exit(1);
	}

	for (int i = 0; i < nConnections; ++i) {
		neuron.weights[i] = rand() / (float)RAND_MAX; // 0~1随机数
	}

	neuron.inputs = (float*)malloc(sizeof(float) * nConnections);
	if (!neuron.inputs) {
		exit(1);
	}

	return neuron;
}

static float sigmoid(float input) {
	return 1.f / (1.f + expf(-input));
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

float neuron_forward(TDNeuron* neuron, float* input) {
	memcpy(neuron->inputs, input, sizeof(float) * neuron->nConnections);
	
	float sum = 0;

//#pragma omp parallel
//#pragma omp for

	for (int i = 0; i < neuron->nConnections; ++i) {
		sum += neuron->weights[i] * input[i] / 16;
	}
	
	neuron->activation = activate(sum);
	neuron->derivate = derivative(sum);

	return neuron->activation;
}

// mse, sigmoid的反向传播
// 迭代过程 https://www.zhihu.com/question/24827633
float* backward(TDNeuron * neuron, float loss, float learningRate)
{
	float* deltaGradients = (float*)malloc(sizeof(float) * neuron->nConnections);
	if (!deltaGradients) {
		exit(1);
	}

	for (int i = 0; i < neuron->nConnections; ++i) {
		deltaGradients[i] = loss * neuron->weights[i] * neuron->derivate;		
		neuron->weights[i] -= learningRate * loss * neuron->inputs[i];
	}

	return deltaGradients;
}

