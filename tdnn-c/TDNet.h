#ifndef TDNET_H_
#define TDNET_H_

#include "TDLayer.h"

typedef struct {	
	float learningRate;					// ѧϰ��
	float decayRate;					// ѧϰ��˥����	
	TDLayer* layers;					// ������
	unsigned int layersCount;			// �������
	float* inputFrames;					// ��ʱ���ڶ�֡����
	unsigned int inputFramesSize;		// ��ʱ���������ݵ�����
	unsigned int input_w;				// �������ݿ��
	unsigned int input_h;				// �������ݸ߶�
	unsigned int input_c;				// ��������ͨ����

} TDNet;

TDNet createTDNet(float learningRate, float decayRate);
void addTDLayer(TDNet *net, const TDLayer *layer);
float* forward(TDNet *net, float* input);

#endif  /* TDNET_H_ */