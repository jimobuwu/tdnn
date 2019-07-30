#ifndef TDLAYER_H_
#define TDLAYER_H_

#include "TDNeuron.h"

typedef struct TDLayer {
	TDNeuron* neurons;					// 神经元数据
	unsigned int neuronsCount;			// 神经元数量
	unsigned int delay;					// 延时数量
	unsigned int inputSize;				// 输入数据尺寸
	float *inputFrames;					// 延时窗口内的数据
	unsigned int inputFramesSize;		// 延时窗口内数据的总量

} TDLayer;

TDLayer createTDLayer(int neuronsCount, int delay, int inputSize);
void layer_forward(TDLayer *layer, float* input, float* output);

#endif /* TDLAYER_H_ */