#ifndef TDNEURON_H_
#define TDNEURON_H_

#include "TDUtils.h"
#include "Macro.h"

typedef struct {
	float epsilon;
	float gamma;
	float mean;
	float var;
	float scale;
	float offset;
} TDNeuronBatchNorm;

typedef struct {

} TDNeuronNorm;

typedef struct {
	float *weights;					// 连接的权重参数
	float bias;						// 偏移量
	TDShape* kernelShape;			// 卷积核的尺寸	
	unsigned int stride_h;			// 高度方向的步长
	float *activation;				// 激活值，由于h方向上有卷积，激活值是二维数组
	unsigned int heightOut;	    // 输出的高度
	int* timeOffsets;				// 选取的上一层输入的偏移量
	ACTIVATION_TYPE actType;		// 激活函数类型
	TDNeuronBatchNorm *bn;			// BN参数
	NORM_TYPE norm_type;			// 规范化类型
	_Bool actBeforeNorm;			// 激活是否在norm前
	float norm_target_rms;			// normalize参数
} TDNeuron;

TDNeuron createTDNeuron(
	ACTIVATION_TYPE actType, 
	const TDShape *kernelShape, 
	const int *timeOffsets,
	unsigned int offsetsSize, 
	unsigned int heightOut,
	_Bool actBeforeNorm);
void neuron_forward(TDNeuron* neuron, float* input, const TDShape *inputShape);
void addNeuronBN(TDNeuron * neuron, float epsilon, float gamma, float mean, float var);

#endif /* TDNEURON_H_ */