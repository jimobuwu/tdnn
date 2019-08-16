#ifndef TDLAYER_H_
#define TDLAYER_H_

#include "TDNeuron.h"
#include "Macro.h"

typedef struct {
	unsigned int id;					// ����	
	LAYER_TYPE type;					// ������
	TDNeuron* neurons;					// ��Ԫ����
	unsigned int neuronsCount;			// ��Ԫ����
	TDShape *input_shape;				// �������ݳߴ�
	float *inputFrames;					// ��ʱ�����ڵ�����
	unsigned int inputFramesSize;		// ��ʱ���������ݵ�����
	unsigned int delay;					// ��ʱ

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