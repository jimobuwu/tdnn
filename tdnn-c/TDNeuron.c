#include "TDNeuron.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "TDUtils.h"

TDNeuron createTDNeuron(unsigned int nConnections, unsigned int kernel_w, unsigned int kernel_h) {
	TDNeuron neuron;
	neuron.nConnections = nConnections;
	neuron.weights = (float*)malloc(sizeof(float) * nConnections);
	neuron.activation = 0.f;
	neuron.kernel_w = kernel_w;
	neuron.kernel_h = kernel_h;

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

static float relu(float input) {

}

static float sigmoid(float input) {
	return 1.f / (1.f + expf(-input));
}

static float activate(float input) {
	//return sigmoid(input);
	return relu(input);
}

float *neuron_forward(TDNeuron* neuron, float* input, unsigned int input_w, unsigned int input_h) {
	memcpy(neuron->inputs, input, sizeof(float) * neuron->nConnections);
	
	float sum = 0;
	
	/*for (int i = 0; i < neuron->nConnections; ++i) {
		sum += neuron->weights[i] * input[i] / 16;
	}*/
	
	unsigned int fm_size = 0;
	float* fm = getConv(input, input_w, input_h, neuron->weights, neuron->kernel_w, neuron->kernel_h, 1, 1, VALID, 0, 0, &fm_size);
	for (int i = 0; i < fm_size; ++i) {

	}


	neuron->activation = activate(sum);

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

#pragma omp parallel for
	for (int i = 0; i < neuron->nConnections; ++i) {
		deltaGradients[i] = loss * neuron->weights[i] * neuron->derivate;		
		neuron->weights[i] -= learningRate * loss * neuron->inputs[i];
	}

	return deltaGradients;
}

