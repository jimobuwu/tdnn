#ifndef TDLAYER_H_
#define TDLAYER_H_

#include "TDNeuron.h"
#include "Macro.h"

typedef struct {
	unsigned int id;					// ����
	unsigned int kernel_w;				// ����˿��
	unsigned int kernel_h;				// ����˸߶�
	TDNeuron* neurons;					// ��Ԫ����
	unsigned int neuronsCount;			// ��Ԫ����
	unsigned int delay;					// ��ʱ����
	unsigned int inputSize;				// �������ݳߴ�
	unsigned int input_w;				// �������ݿ��
	unsigned int input_h;				// �������ݸ߶�
	float *inputFrames;					// ��ʱ�����ڵ�����
	unsigned int inputFramesSize;		// ��ʱ���������ݵ�����
	LAYER_TYPE type;					// ������

} TDLayer;

TDLayer createTDLayer(unsigned int id, unsigned int neuronsCount, unsigned int delay, unsigned int inputSize);
void layer_forward(TDLayer *layer, float* input, float* output);

#endif /* TDLAYER_H_ */