#ifndef TDLAYER_H_
#define TDLAYER_H_

#include "TDNeuron.h"

typedef struct {
	unsigned int id;					// ����
	TDNeuron* neurons;					// ��Ԫ����
	unsigned int neuronsCount;			// ��Ԫ����
	unsigned int delay;					// ��ʱ����
	unsigned int inputSize;				// �������ݳߴ�
	float *inputFrames;					// ��ʱ�����ڵ�����
	unsigned int inputFramesSize;		// ��ʱ���������ݵ�����
} TDLayer;

TDLayer createTDLayer(unsigned int id, unsigned int neuronsCount, unsigned int delay, unsigned int inputSize);
void layer_forward(TDLayer *layer, float* input, float* output);

#endif /* TDLAYER_H_ */