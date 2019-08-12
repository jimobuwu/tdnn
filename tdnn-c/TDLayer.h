#ifndef TDLAYER_H_
#define TDLAYER_H_

#include "TDNeuron.h"
#include "Macro.h"

typedef struct {
	unsigned int id;					// 层数
	unsigned int kernel_w;				// 卷积核宽度
	unsigned int kernel_h;				// 卷积核高度
	TDNeuron* neurons;					// 神经元数据
	unsigned int neuronsCount;			// 神经元数量
	unsigned int delay;					// 延时数量
	unsigned int inputSize;				// 输入数据尺寸
	unsigned int input_w;				// 输入数据宽度
	unsigned int input_h;				// 输入数据高度
	float *inputFrames;					// 延时窗口内的数据
	unsigned int inputFramesSize;		// 延时窗口内数据的总量
	LAYER_TYPE type;					// 层类型

} TDLayer;

TDLayer createTDLayer(unsigned int id, unsigned int neuronsCount, unsigned int delay, unsigned int inputSize);
void layer_forward(TDLayer *layer, float* input, float* output);

#endif /* TDLAYER_H_ */