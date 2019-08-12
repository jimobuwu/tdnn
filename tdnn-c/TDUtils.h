#ifndef TDUTILS_H_
#define TDUTILS_H_

#include "Macro.h"

float* getConv(const float* input, unsigned int input_width, unsigned int input_height,
	const float* conv, unsigned int kernel_width, unsigned int kernel_height,
	unsigned int stride_row, unsigned int stride_col, PADDING_TYPE pad_type, unsigned int padding_w, unsigned int padding_h, unsigned int *fm_size);

#endif

