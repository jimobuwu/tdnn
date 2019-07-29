#include "TDNet.h"


TDNet createTDNet() {
	TDNet net;
	net.layersCount = 3;
	net.learningRate = 0.01;
	net.decayRate = 0.9;
	net.inputDelay = 2;
	net.inputSize = 16;
	net.layers = (TDLayer*)malloc(sizeof(TDLayer) * net.layersCount);
	net.inputFrames = (float*)malloc(sizeof(float) * (net.inputDelay + 1) * net.inputSize);

	net.layers[0] = createTDLayer(8, 2, 16); // 隐藏层1
	net.layers[1] = createTDLayer(3, 4, 8);  // 隐藏层2
	net.layers[2] = createTDLayer(1, 0, 3);  //输出层

	return net;
}

void pushFrame(TDNet *net, float *input, int inputSize) {
	// 从第二帧开始前移
	for (int i = 0; i < net->inputDelay; ++i) {
		memcpy(&net->inputFrames[(i + 1) * inputSize], &net->inputFrames[i * inputSize], inputSize);
	}

	// 加入新帧
	memcpy(&net->inputFrames[net->inputDelay * inputSize], input, inputSize);
}

// target, 三个辅音的得分
void train(TDNet *net, float *target) {
	float* output = (float*)malloc(sizeof(float) * net->layers[net->layersCount - 1].neuronsCount)

	// 前向
	for (int i = 0; i < net->layersCount; ++i) {
		TDLayer layer = net->layers[i];

		double *activation = (float*)malloc(sizeof(float) * layer.neuronsCount);
		for (int j = 0; j < layer.neuronsCount; ++j) {
			activation[j] = forward(&layer.neurons[j], net->inputFrames);

		}
	}

	// 反向
}



