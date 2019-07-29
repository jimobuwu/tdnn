#ifndef TDNET_H_
#define TDNET_H_

#include "TDLayer.h"

typedef struct TDNet {
	int delay;					//��ʱ����
	float learningRate;
	float decayRate;
	TDLayer* layers;
	int layersCount;
	
} TDNet;

TDNet createTDNet();
void train(TDNet net, float *trainData);

#endif  /* TDNET_H_ */