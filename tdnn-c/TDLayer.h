#ifndef TDLAYER_H_
#define TDLAYER_H_

#include "TDNeuron.h"
#include "Macro.h"

typedef struct {
	unsigned int id;					// 层数	
	LAYER_TYPE type;					// 层类型
	TDNeuron* neurons;					// 神经元数据
	unsigned int neuronsCount;			// 神经元数量
	TDShape *input_shape;				// 输入数据尺寸
	float *inputFrames;					// 延时窗口内的数据
	unsigned int inputFramesSize;		// 延时窗口内数据的总量
	unsigned int delay;					// 延时

} TDLayer;

TDLayer createTDLayer(
	unsigned int id, 
	LAYER_TYPE layer_type, 
	unsigned int neuronsCount, 
	const TDShape *kernel_shape,
	const float *time_offsets, 
	unsigned int offsets_size, 
	const TDShape *input_shape,
	unsigned int height_out);

void layer_forward(TDLayer* layer, float* input, float* output);

#endif /* TDLAYER_H_ */