#include "TDUtils.h"
#include "assert.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

float * getConv(const float * input, unsigned int input_width, unsigned int input_height,
	const float * conv, unsigned int kernel_width, unsigned int kernel_height,
	unsigned int stride_row, unsigned int stride_col,
	PADDING_TYPE pad_type, unsigned int padding_w, unsigned int padding_h,
	unsigned int *fm_size)
{
	assert(input_width > kernel_width);
	assert(input_height > kernel_height);

	int input_w_new = input_width;
	int input_h_new = input_height;

	int out_w = 0;
	int out_h = 0;

	int padding_w_l = 0;
	int padding_w_r = 0;
	int padding_h_l = 0;
	int padding_h_r = 0;

	float *input_new = NULL;

	if ( SAME == pad_type ) {
		if (1 == stride_row) {
			padding_w_l = padding_w_r = ceil((kernel_width - 1) / 2);
			padding_w = padding_w_l + padding_w_r;
			out_w = input_width;
		}
		else {
			// 可以整除步长的最小整数padding
			padding_w = ceil(ceil((input_width - kernel_width) / stride_row) * stride_row - (input_width - kernel_width));

			if (0 == padding_w & 1) {
				// 偶数padding，左右一半
				padding_w_l = padding_w_r = padding_w / 2;
			}
			else {
				// 奇数padding，左奇右偶
				padding_w_l = ceil(padding_w / 2);
				padding_w_r = padding_w - padding_w_l;
			}

			out_w = (input_width - kernel_width + padding_w) / stride_row + 1;
		}

		if (1 == stride_col) {
			padding_h = ceil((kernel_height - 1) / 2);
			padding_h = padding_h_l + padding_h_r;
			out_h = input_height;
		}
		else {
			padding_h = ceil(ceil((input_height - kernel_height) / stride_col) * stride_col - (input_height - kernel_height));
			if (0 == padding_h & 1) {
				padding_h_l = padding_h_r = padding_h / 2;
			}
			else {
				padding_h_l = ceil(padding_h / 2);
				padding_h_r = padding_h - padding_h_l;
			}
			out_h = (input_height - kernel_height + padding_h) / stride_col + 1;
		}

		input_w_new += padding_w;
		input_h_new += padding_h;

		int len_new = input_w_new * input_h_new;
		input_new = (float*)calloc(len_new, sizeof(float));
		int index = 0;
		for (int i = padding_h_l; i < input_h_new - padding_h_r; ++i) {
			for (int j = padding_w_l; j < input_w_new - padding_w_r; ++j) {
				input_new[i * input_w_new + j] = input[index];
				++index;
			}
		}
	}
	else {
		// N = ( W - F + 2P ) / S + 1
		out_w = (input_width - kernel_width + 2 * padding_w) / stride_row + 1;
		out_h = (input_height - kernel_height + 2 * padding_h) / stride_col + 1;

		int len_new = input_w_new * input_h_new;
		input_new = (float*)malloc(sizeof(float) * len_new);
		if (!input_new) {
			abort();
		}
		memcpy(input_new, input, sizeof(float) * len_new);
	}

	const int conv_len = kernel_width * kernel_height;
	float* tmp = (float*)calloc(out_w * out_h * conv_len, sizeof(float));

	// flatten and save
	int wh = 0;
	for (int i = 0; i < out_h; ++i) {
		for (int j = 0; j < out_w; ++j) {
			wh = (i * out_w + j) * conv_len;

			int start = 0;
			for (int m = 0; m < kernel_height; ++m) {
				start = (i * stride_col + m) * input_w_new + j * stride_row;

				for (int n = 0; n < kernel_width; ++n) {
					if ( SAME == pad_type )
						tmp[wh + m * kernel_width + n] = input_new[start + n];
					else
						tmp[wh + m * kernel_width + n] = input[start + n];
				}
			}
		}
	}

	float* out = (float*)malloc(sizeof(float) * out_w * out_h);
	for (int i = 0; i < out_w; ++i) {
		for (int j = 0; j < out_h; ++j) {
			int conv_value = 0;
			wh = (i * out_h + j) * conv_len;
			for (int m = 0; m < conv_len; ++m) {
				conv_value += tmp[wh + m] * conv[m];
			}
			out[i * out_h + j] = conv_value;
		}
	}

	return out;
}
