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
	unsigned int inputDelay;			// ������ʱ
	unsigned int inputSize;				// ����һ֡�ĳߴ�
	
} TDNet;

TDNet createTDNet();
float* train(TDNet *net, float *trainData);

#endif  /* TDNET_H_ */