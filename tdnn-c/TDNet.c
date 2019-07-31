#include "TDNet.h"
#include <stdlib.h>
#include <stdio.h>
#include "Macro.h"
#include <string.h>

TDNet createTDNet() {
	TDNet net;
	net.layersCount = 3;
	net.learningRate = 0.01f;
	net.decayRate = 0.9f;
	net.inputDelay = 2;
	net.inputSize = 16;
	net.layers = (TDLayer*)malloc(sizeof(TDLayer) * net.layersCount);
	if (!net.layers) {
		fprintf(stderr, "createTDNet malloc layers fail!");
		exit(1);
	}

	net.inputFramesSize = (net.inputDelay + 1) * net.inputSize;
	net.inputFrames = (float*)calloc(net.inputFramesSize, sizeof(float));
	if (!net.inputFrames) {
		fprintf(stderr, "createTDNet malloc inputFrames fail!");
		exit(1);
	}

	net.layers[0] = createTDLayer(8, 2, 16); // 隐藏层1 3 * 16 -> 8
	net.layers[1] = createTDLayer(3, 4, 8);  // 隐藏层2 5 * 8 -> 3

	// 输出层   9 * 3 -> 3 
	// 论文中输出层的计算是计算一个神经元9帧的求和。
	// 但同时也指出可以使用sigmoid计算，没太明白。
	// 三个输出节点的权重相同？参考论文 330
	// 先按照输出层是3个神经元组成，每个神经元有3 * 9 个权重，输入是由隐藏层2中3个神经元，9帧求和后，sigmoid激活。
	net.layers[2] = createTDLayer(3, 8, 3);  

	return net;
}

// target, 三个辅音的得分
float* train(TDNet *net, float *target) {
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
		free(activation); 
	}
	free(passData);
	
	// BP, 迭代过程https://www.zhihu.com/question/24827633
	unsigned int nCount = net->layers[net->layersCount - 1].neuronsCount;
	float *loss = (float*)malloc(sizeof(float) * nCount);
	if (!loss) {
		fprintf(stderr, "in backpropagation, malloc loss fail!");
		exit(1);
	}

	// 哪种方式更好？
	 //float *loss = malloc(sizeof(*loss) * nCount);

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
			// 下一层所有连接的delta * w的和， 用于计算上一层w的梯度
			for (int k = 0; k < layer->inputFramesSize; ++k)
				errorSum[k] += gradients[k];
			free(gradients);
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

//float inference()


