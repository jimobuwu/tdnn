#ifndef TDLAYER_H_
#define TDLAYER_H_

#include "TDNeuron.h"

typedef struct TDLayer {
	TDNeuron* neurons;
	int neuronsCount;
	int delay;					//ÑÓÊ±ÊýÁ¿
	int inputSize;

} TDLayer;

TDLayer createTDLayer(int neuronsCount, int delay, int inputSize);

#endif /* TDLAYER_H_ */