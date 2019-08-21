#ifndef TDNEURON_H_
#define TDNEURON_H_

#include "TDUtils.h"
#include "Macro.h"

typedef struct {
	float *weights;					// ���ӵ�Ȩ�ز���
	float bias;						// ƫ����
	TDShape* kernel_shape;			// ����˵ĳߴ�	
	unsigned int stride_h;			// �߶ȷ���Ĳ���
	float *activation;				// ����ֵ������h�������о��������ֵ�Ƕ�ά����

	unsigned int height_out;	    // ����ĸ߶�

	int* time_offsets;			// ѡȡ����һ�������ƫ����
	ACTIVATION_TYPE act_type;		// ���������

	// bn
	 //has_bn;
	float bn_mean;
	float bn_var;
	float bn_epsilon;

} TDNeuron;

TDNeuron createTDNeuron(
	ACTIVATION_TYPE act_type, 
	const TDShape *kernel_shape, 
	const int *time_offsets,
	unsigned int offsets_size, 
	unsigned int height_out);

void neuron_forward(TDNeuron* neuron, float* input, const TDShape *input_shape);

#endif /* TDNEURON_H_ */