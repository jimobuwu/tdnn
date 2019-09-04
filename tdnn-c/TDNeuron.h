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
	float *weights;					// ���ӵ�Ȩ�ز���
	float bias;						// ƫ����
	TDShape* kernelShape;			// ����˵ĳߴ�	
	unsigned int stride_h;			// �߶ȷ���Ĳ���
	float *activation;				// ����ֵ������h�������о��������ֵ�Ƕ�ά����
	unsigned int heightOut;	    // ����ĸ߶�
	int* timeOffsets;				// ѡȡ����һ�������ƫ����
	ACTIVATION_TYPE actType;		// ���������
	TDNeuronBatchNorm *bn;			// BN����
	NORM_TYPE norm_type;			// �淶������
	_Bool actBeforeNorm;			// �����Ƿ���normǰ
	float norm_target_rms;			// normalize����
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