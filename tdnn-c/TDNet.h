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
	unsigned int input_w;				// 输入数据宽度
	unsigned int input_h;				// 输入数据高度
	unsigned int input_c;				// 输入数据通道数

} TDNet;

TDNet createTDNet(float learningRate, float decayRate);
void addTDLayer(TDNet *net, const TDLayer *layer);
float* forward(TDNet *net, float* input);

#endif  /* TDNET_H_ */