#include "TDNet.h"
#include <stdlib.h>

TDNet createTDNet() {
	TDNet net;
	net.layersCount = 3;
	net.learningRate = 0.01f;
	net.decayRate = 0.9f;
	net.inputDelay = 2;
	net.inputSize = 16;
	net.layers = (TDLayer*)malloc(sizeof(TDLayer) * net.layersCount);
	net.inputFramesSize = (net.inputDelay + 1) * net.inputSize;
	net.inputFrames = (float*)calloc(net.inputFramesSize, sizeof(float));

	net.layers[0] = createTDLayer(8, 2, 16); // ���ز�1 3 * 16 -> 8
	net.layers[1] = createTDLayer(3, 4, 8);  // ���ز�2 5 * 8 -> 3

	// �����   9 * 3 -> 3 
	// �����������ļ����Ǽ���һ����Ԫ9֡����͡�
	// ��ͬʱҲָ������ʹ��sigmoid���㣬û̫���ס�
	// ��������ڵ��Ȩ����ͬ���ο����� 330
	// �Ȱ����������3����Ԫ��ɣ�ÿ����Ԫ��3 * 9 ��Ȩ�أ������������ز�2��3����Ԫ��9֡��ͺ�sigmoid���
	net.layers[2] = createTDLayer(3, 8, 3);  

	return net;
}

// target, ���������ĵ÷�
float* train(TDNet *net, float *target) {
	float *output = (float*)malloc(sizeof(float) * net->layers[net->layersCount - 1].neuronsCount);
	float *passData = (float*)malloc(sizeof(float) * net->inputFramesSize);	

	// ǰ��
	for (int i = 0; i < net->layersCount; ++i) {
		TDLayer *layer = &net->layers[i];
		double *activation = (float*)malloc(sizeof(float) * layer->neuronsCount);
		layer_forward(&layer, passData, activation);
		// ���һ�����
		if (i == net->layersCount - 1)
			memcpy(output, activation, sizeof(float) * layer->neuronsCount);

		// ��һ��ļ���ֵ��Ϊ��һ�������
		passData = (float*)realloc(passData, sizeof(float) * layer->neuronsCount);
		memcpy(passData, activation, sizeof(float) * layer->neuronsCount);
		free(activation); 
	}
	free(passData);
	
	// BP, ��������https://www.zhihu.com/question/24827633
	unsigned int nCount = net->layers[net->layersCount - 1].neuronsCount;
	float *loss = (float*)malloc(sizeof(float) * nCount);
	for (int i = 0; i < nCount; ++i) {
		loss[i] = output[i] - target[i];
	}

	for (int i = net->layersCount - 1; i >= 0; --i) {
		TDLayer *layer = &net->layers[i];		
		float *errorSum = (float*)calloc(layer->inputFramesSize, sizeof(float));

		for (int j = 0; j < layer->neuronsCount; ++j) {
			float *gradients = backward(&layer->neurons[j], loss[j], net->learningRate);
			// ��һ���������ӵ�delta * w�ĺͣ� ���ڼ�����һ��w���ݶ�
			for (int k = 0; k < layer->inputFramesSize; ++k)
				errorSum[k] += gradients[k];
			free(gradients);
		}
		loss = (float*)realloc(loss, sizeof(float) * layer->inputFramesSize);
		memcpy(loss, errorSum, sizeof(float) * layer->inputFramesSize);
		free(errorSum);
	}
	net->learningRate *= net->decayRate;
	free(loss);

	return output;
}



