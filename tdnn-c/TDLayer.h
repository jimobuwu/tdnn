#ifndef TDLAYER_H_
#define TDLAYER_H_

#include "TDNeuron.h"
#include "Macro.h"
#include "TDUtils.h"

typedef struct {
	unsigned int id;					// 层数	
	char *name;							// 名字，和模型文件的component对应
	LAYER_TYPE type;					// 层类型
	TDNeuron *neurons;					// 神经元数据
	unsigned int neuronsCount;			// 神经元数量
	TDShape *input_shape;				// 输入数据尺寸
	float *inputFrames;					// 延时窗口内的数据
	unsigned int inputFramesSize;		// 延时窗口内数据的总量
	unsigned int delay;					// 延时
	unsigned int curBufferFrameSize;    // 当前缓存的帧数
	_Bool has_logsoftmax;
	_Bool is_output;

} TDLayer;

TDLayer createTDLayer(
	unsigned int id,
	const char* name,
	LAYER_TYPE layer_type, 
	ACTIVATION_TYPE act_type,
	unsigned int neuronsCount, 
	const TDShape *kernel_shape,
	const int *time_offsets, 
	unsigned int offsets_size, 
	const TDShape *input_shape,
	unsigned int height_out,	
	_Bool has_logsoftmax);

int layer_forward(TDLayer *layer, const float *input, float *output);
void load_weights(TDLayer *layer, const char *filePath);
void addBN(TDLayer *layer, const char* filePath, unsigned dim, float epsilon, unsigned count, float gamma);

//void load_relu_weights(TDLayer *layer, const float *weights);
//void load_bn_weights(TDLayer *layer, const float *weights);

#endif /* TDLAYER_H_ */