#include "TDNet.h"
#include <stdlib.h>
#include <stdio.h>
#include "Macro.h"
#include <string.h>
#include <omp.h>

TDNet createTDNet() {
	TDNet net;
	net.layersCount = 3;
	net.learningRate = 0.01f;
	net.decayRate = 0.9f;
	net.layers = (TDLayer*)malloc(sizeof(TDLayer) * net.layersCount);
	if (!net.layers) {
		fprintf(stderr, "createTDNet malloc layers fail!");
		exit(1);
	}

	//net.inputFramesSize = (net.inputDelay + 1) * net.inputSize;
	net.inputFrames = (float*)calloc(net.inputFramesSize, sizeof(float));
	if (!net.inputFrames) {
		fprintf(stderr, "createTDNet malloc inputFrames fail!");
		exit(1);
	}

	net.layers[0] = createTDLayer(0, 8, 2, 16); // 隐藏层1 3 * 16 -> 8
	net.layers[1] = createTDLayer(1, 3, 4, 8);  // 隐藏层2 5 * 8 -> 3
	net.layers[2] = createTDLayer(2, 3, 8, 3);  

	return net;
}

// target, 三个辅音的得分
float* train(TDNet *net, float* input, float *target) {
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
		float *activation = (float*)malloc(sizeof(float) * layer->neuronsCount);
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
	
	// BP, 迭代过程 https://www.zhihu.com/question/24827633
	unsigned int nCount = net->layers[net->layersCount - 1].neuronsCount;
	float *loss = (float*)malloc(sizeof(float) * nCount);
	if (!loss) {
		fprintf(stderr, "in backpropagation, loss malloc fail!");
		exit(1);
	}

	#pragma omp parallel for
	for (int i = 0; i < nCount; ++i) {
		loss[i] = output[i] - target[i];
	}

	for (unsigned int i = net->layersCount - 1; i >= 0; --i) {
		TDLayer *layer = &net->layers[i];		
		float *errorSum = (float*)calloc(layer->inputFramesSize, sizeof(float));
		if (!errorSum) {
			fprintf(stderr, "in backpropagation, layer %d calloc errorSum fail!", i);
			exit(1);
		}

		for (int j = 0; j < layer->neuronsCount; ++j) {
			float *gradients = backward(&layer->neurons[j], loss[j], net->learningRate);
			// 下一层所有连接的 delta * w 的和， 用于计算上一层w的梯度
			for (int k = 0; k < layer->inputFramesSize; ++k)
				errorSum[k] += gradients[k];
			SAFEFREE(gradients);
		}
		loss = (float*)realloc(loss, sizeof(float) * layer->inputFramesSize);
		if (!loss) {
			fprintf(stderr, "in backpropagation, layer %d realloc loss fail!", i);
			exit(1);
		}

		memcpy(loss, errorSum, sizeof(float) * layer->inputFramesSize);
		SAFEFREE(errorSum);
	}
	net->learningRate *= net->decayRate;
	SAFEFREE(loss);

	return output;
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
		float *activation = (float*)malloc(sizeof(float) * layer->neuronsCount);
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


