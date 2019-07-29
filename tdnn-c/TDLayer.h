#ifndef TDLAYER_H_
#define TDLAYER_H_

#include "TDNeuron.h"

typedef struct TDLayer {
	struct TDNeuron* neurons;
	int neuronsCount;

} TDLayer;

TDLayer createTDLayer(int neuronsCount);

#endif /* TDLAYER_H_ */