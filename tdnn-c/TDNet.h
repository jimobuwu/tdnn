#ifndef TDNET_H_
#define TDNET_H_

#include "TDLayer.h"

typedef struct {	
	float learningRate;					// 学习率
	float decayRate;					// 学习率衰减率
	TDLayer* layers;					// 层数据
	unsigned int layersCount;			// 网络层数
	float* inputFrames;					// 延时窗口多帧数据
	unsigned int inputFramesSize;		// 延时窗口内数据的总量
	unsigned int inputDelay;			// 输入延时
	unsigned int inputSize;				// 输入一帧的尺寸
	
} TDNet;

TDNet createTDNet();
float* train(TDNet *net, float *trainData);

#endif  /* TDNET_H_ */