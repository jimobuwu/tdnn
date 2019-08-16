#include "TDUtils.h"
#include "assert.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>


float* getConv(const float *input, const TDShape *input_shape,
	const float *kernel, const TDShape *kernel_shape, unsigned int stride_h) {

	//assert(input_shape->w > kernel_shape->w);
	//assert(input_shape->h > kernel_shape->h);

	int out_w = 1;
	int out_h = (input_shape->h - kernel_shape->h) / stride_h + 1;
		
	const int conv_len = kernel_shape->w * kernel_shape->h * kernel_shape->c;
	float* tmp = (float*)calloc(out_w * out_h * conv_len, sizeof(float));
	if (!tmp) {
		abort();
	}

	// flatten and save

	// out_h = 9
	// out_w = 1
	// kernel_shape->w = 8
	// kernel_shape->h = 18
	// kernel_shape->c = 3

	int input_w = input_shape->h * input_shape->c;

	int wh = 0;
	int index = 0;
	for (int i = 0; i < out_h; ++i) {
		for (int j = 0; j < out_w; ++j) {
			for (int c = 0; c < kernel_shape->c; ++c) {
				int start = input_w * i + input_shape->h * c;

				for (int w = 0; w < kernel_shape->h; ++w) {
					tmp[index++] = input[start + w];
				}
			}
		}
	}

	float* out = (float*)malloc(sizeof(float) * out_w * out_h);
	if (!out) {
		abort();
	}

	for (int i = 0; i < out_h; ++i) {
		for (int j = 0; j < out_w; ++j) {
			int conv_value = 0;
			wh = (i * out_w + j) * conv_len;
			for (int m = 0; m < conv_len; ++m) {
				conv_value += tmp[wh + m] * kernel[m];
			}
			out[i * out_w + j] = conv_value;
		}
	}

	return out;
}
