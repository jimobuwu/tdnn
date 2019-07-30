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

	net.layers[0] = createTDLayer(8, 2, 16); // ���ز�1
	net.layers[1] = createTDLayer(3, 4, 8);  // ���ز�2
	net.layers[2] = createTDLayer(1, 0, 3);  //�����

	return net;
}

//void pushFrame(TDNet *net, float *input, int inputSize) {
//	// �ӵڶ�֡��ʼǰ��
//	for (int i = 0; i < net->inputDelay; ++i) {
//		memcpy(&net->inputFrames[(i + 1) * inputSize], &net->inputFrames[i * inputSize], inputSize);
//	}
//
//	// ������֡
//	memcpy(&net->inputFrames[net->inputDelay * inputSize], input, inputSize);
//}

// target, ���������ĵ÷�
void train(TDNet *net, float *target) {
	float *output = (float*)malloc(sizeof(float) * net->layers[net->layersCount - 1].neuronsCount);
	float *passData = (float*)malloc(sizeof(float) * net->inputFramesSize);	

	// ǰ��
	for (int i = 0; i < net->layersCount; ++i) {
		TDLayer layer = net->layers[i];
		double *activation = (float*)malloc(sizeof(float) * layer.neuronsCount);
		layer_forward(&layer, passData, activation);
		// ���һ�����
		if (i == net->layersCount - 1)
			memcpy(output, activation, sizeof(float) * layer.neuronsCount);

		// ��һ��ļ���ֵ��Ϊ��һ�������
		passData = (float*)realloc(passData, sizeof(float) * layer.neuronsCount);
		memcpy(passData, activation, sizeof(float) * layer.neuronsCount);
		free(activation); 
	}
	free(passData);
	
	// ����

}



