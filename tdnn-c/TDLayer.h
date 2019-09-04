#ifndef TDLAYER_H_
#define TDLAYER_H_

#include "TDNeuron.h"
#include "Macro.h"
#include "TDUtils.h"

typedef struct {
	unsigned int id;					// 层数	
	char *name;							// 名字，和模型文件的component对应
	layerType type;						// 层类型
	TDNeuron *neurons;					// 神经元数据
	unsigned int neuronsCount;			// 神经元数量
	TDShape *inputShape;				// 输入数据尺寸
	float *inputFrames;					// 延时窗口内的数据
	unsigned int inputFramesSize;		// 延时窗口内数据的总量
	unsigned int delay;					// 延时
	unsigned int curBufferFrameSize;    // 当前缓存的帧数
	_Bool hasLogsoftmax;				// 是否调用LogSoftMax 
	_Bool isOutput;						// 是否是输出层	
	_Bool hasLN;						// 是否有layer normalization
	float LNTargetRms;					// layer normalization target rms
		
} TDLayer;

TDLayer createTDLayer(
	unsigned int id,
	const char* name,
	layerType layerType, 
	ACTIVATION_TYPE actType,
	unsigned int neuronsCount, 
	const TDShape *kernelShape,
	const int *timeOffsets, 
	unsigned int offsetsSize, 
	const TDShape *inputShape,
	unsigned int heightOut,	
	_Bool hasLogsoftmax,
	_Bool actBeforeNorm);

int layer_forward(TDLayer *layer, const float *input, float *output, const char *outputFilePath);
void load_weights(TDLayer *layer, const char *filePath);
void addBN(TDLayer *layer, const char* filePath, unsigned dim, float epsilon, float gamma);
void addLayerNorm(TDLayer *layer, float targetRms);

#endif /* TDLAYER_H_ */