#ifndef TDLAYER_H_
#define TDLAYER_H_

#include "TDNeuron.h"
#include "Macro.h"
#include "TDUtils.h"

typedef struct {
	unsigned int id;					// ����	
	char *name;							// ���֣���ģ���ļ���component��Ӧ
	LAYER_TYPE type;					// ������
	TDNeuron *neurons;					// ��Ԫ����
	unsigned int neuronsCount;			// ��Ԫ����
	TDShape *input_shape;				// �������ݳߴ�
	float *inputFrames;					// ��ʱ�����ڵ�����
	unsigned int inputFramesSize;		// ��ʱ���������ݵ�����
	unsigned int delay;					// ��ʱ
	unsigned int curBufferFrameSize;    // ��ǰ�����֡��
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