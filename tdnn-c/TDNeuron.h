#ifndef TDNEURON_H_
#define TDNEURON_H_

#include "TDUtils.h"
#include "Macro.h"

typedef struct {
	float epsilon;
	unsigned count;
	float gamma;
	float mean;
	float var;
	float scale;
	float offset;
} TDNeuronBatchNorm;

typedef struct {
	float *weights;					// 连接的权重参数
	float bias;						// 偏移量
	TDShape* kernel_shape;			// 卷积核的尺寸	
	unsigned int stride_h;			// 高度方向的步长
	float *activation;				// 激活值，由于h方向上有卷积，激活值是二维数组

	unsigned int height_out;	    // 输出的高度

	int* time_offsets;				// 选取的上一层输入的偏移量
	ACTIVATION_TYPE act_type;		// 激活函数类型

	TDNeuronBatchNorm *bn;

} TDNeuron;

TDNeuron createTDNeuron(
	ACTIVATION_TYPE act_type, 
	const TDShape *kernel_shape, 
	const int *time_offsets,
	unsigned int offsets_size, 
	unsigned int height_out);
void neuron_forward(TDNeuron* neuron, float* input, const TDShape *input_shape);
void addNeuronBN(TDNeuron * neuron, float epsilon, unsigned count, float gamma, float mean, float var);

#endif /* TDNEURON_H_ */