#ifndef TDLAYER_H_
#define TDLAYER_H_

#include "TDNeuron.h"
#include "Macro.h"
#include "TDUtils.h"

typedef struct {
	unsigned int id;					// ����	
	char *name;							// ���֣���ģ���ļ���component��Ӧ
	layerType type;						// ������
	TDNeuron *neurons;					// ��Ԫ����
	unsigned int neuronsCount;			// ��Ԫ����
	TDShape *inputShape;				// �������ݳߴ�
	float *inputFrames;					// ��ʱ�����ڵ�����
	unsigned int inputFramesSize;		// ��ʱ���������ݵ�����
	unsigned int delay;					// ��ʱ
	unsigned int curBufferFrameSize;    // ��ǰ�����֡��
	_Bool hasLogsoftmax;				// �Ƿ����LogSoftMax 
	_Bool isOutput;						// �Ƿ��������	
	_Bool hasLN;						// �Ƿ���layer normalization
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