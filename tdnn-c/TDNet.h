#ifndef TDNET_H_
#define TDNET_H_

#include "TDLayer.h"

typedef struct TDNet {	
	float learningRate;
	float decayRate;
	TDLayer* layers;
	int layersCount;
	float* inputFrames;
	int inputDelay;
	int inputSize;
	
} TDNet;

TDNet createTDNet();
void pushFrame(TDNet *net, float* input, int inputSize);
void train(TDNet *net, float *trainData);

#endif  /* TDNET_H_ */