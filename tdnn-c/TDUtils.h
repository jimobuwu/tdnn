#ifndef TDUTILS_H_
#define TDUTILS_H_

#include "Macro.h"

typedef struct {
	unsigned int w;  // weights width
	unsigned int h;  // weights height
	unsigned int c;  // weights channel
} TDShape;

float* getConv(const float *input, const TDShape *inputShape,
	const float *kernel, const TDShape *kernelShape, unsigned int stride_h);

void parseWeights(const char *file, unsigned weights_rows, 
	float *linear_weights, float *bias_weights);

#endif

