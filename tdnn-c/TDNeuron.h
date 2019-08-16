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

	unsigned int height_int;		// ����ĸ߶�
	unsigned int height_out;	    // ����ĸ߶�

	float* time_offsets;			// ѡȡ����һ�������ƫ����
	ACTIVATION_TYPE act_type;		// ���������

}TDNeuron;

TDNeuron createTDNeuron(ACTIVATION_TYPE act_type, const TDShape *kernel_shape, const float *time_offsets, unsigned int offsets_size, unsigned int height_out);
void neuron_forward(TDNeuron* neuron, float* input, const TDShape *input_shape);

#endif /* TDNEURON_H_ */