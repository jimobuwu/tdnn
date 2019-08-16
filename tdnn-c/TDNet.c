#include "TDNet.h"
#include <stdlib.h>
#include <stdio.h>
#include "Macro.h"
#include <string.h>
#include <omp.h>

TDNet createTDNet(float learningRate, float decayRate) {
	TDNet net;
	net.layersCount = 0;
	net.learningRate = learningRate;
	net.decayRate = decayRate;

	return net;
}

void addTDLayer(TDNet *net, const TDLayer *layer) {
	++net->layersCount;
	net->layers = (TDLayer*)realloc(net->layers, sizeof(TDLayer) * net->layersCount);
	if (!net->layers) {
		fprintf(stderr, "createTDNet malloc layers fail!");
		exit(1);
	}
}

float* forward(TDNet * net, float * input){
	float *output = (float*)malloc(sizeof(float) * net->layers[net->layersCount - 1].neuronsCount);
	if (!output) {
		fprintf(stderr, "train malloc output fail!");
		exit(1);
	}

	float *passData = (float*)malloc(sizeof(float) * net->inputFramesSize);
	if (!passData) {
		fprintf(stderr, "train malloc passData fail!");
		exit(1);
	}
	memcpy(passData, input, sizeof(float) * net->inputFramesSize);

	// 前向
	for (int i = 0; i < net->layersCount; ++i) {
		TDLayer *layer = &net->layers[i];
		float *activation = (float*)malloc(sizeof(float) * layer->neuronsCount * layer->neurons[0].height_out);
		if (!activation) {
			fprintf(stderr, "in forward propagation, layer %d malloc activation fail!", i);
			exit(1);
		}

		layer_forward(layer, passData, activation);
		// 最后一层输出
		if (i == net->layersCount - 1)
			memcpy(output, activation, sizeof(float) * layer->neuronsCount);

		// 上一层的激活值作为下一层的输入
		passData = (float*)realloc(passData, sizeof(float) * layer->neuronsCount);
		if (!passData) {
			fprintf(stderr, "in forward propagation, layer %d realloc passData fail!", i);
			exit(1);
		}

		memcpy(passData, activation, sizeof(float) * layer->neuronsCount);
		SAFEFREE(activation);
	}
	SAFEFREE(passData);

	return output;
}


