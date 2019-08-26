#include "TDUtils.h"
#include "assert.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>


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
	//for (int i = 0; i < out_h; ++i) {
	//	for (int j = 0; j < out_w; ++j) {
	//		for (int c = 0; c < kernel_shape->c; ++c) {
	//			int start = input_w * i + input_shape->h * c;

	//			for (int w = 0; w < kernel_shape->h; ++w) {
	//				tmp[index++] = input[start + w];
	//			}
	//		}
	//	}
	//}

	for (int i = 0; i < out_h; ++i) {
		for (int w = 0; w < kernel_shape->w; ++w) {
			for (int c = 0; c < kernel_shape->c; ++c) {
				for (int h = 0; h < kernel_shape->h; ++h) {
					int start = w * input_w + c * input_shape->h + i;
					tmp[index++] = input[start + h];
				}
			}
		}
	}


	float* out = (float*)malloc(sizeof(float) * out_w * out_h);
	if (!out) {
		printf("malloc fail! getConv out ");
		abort();
	}

	for (int i = 0; i < out_h; ++i) {
		for (int j = 0; j < out_w; ++j) {
			float conv_value = 0.f;
			wh = (i * out_w + j) * conv_len;
			for (int m = 0; m < conv_len; ++m) {
				conv_value += tmp[wh + m] * kernel[m];
			}
			out[i * out_w + j] = conv_value;
		}
	}
	   
	return out;
}

void parseWeights(const char *file, unsigned weights_rows, float *linear_weights, float *bias_weights)
{
	FILE *fp = fopen(file, "r");
	if (!fp) {
		return;
	}

	char line[LINE_BUF_SIZE];
	unsigned count = 0;
	int line_num = 0;	

	while (!feof(fp)) {			
		fgets(line, LINE_BUF_SIZE, fp);
		//printf("\n line: %s", line);

		if (line_num >= 1 && line_num <= weights_rows + 1) {
			// linearParams
			char *p = line, *end;			
			for (float f = strtof(p, &end); p != end; f = strtof(p, &end)) {
				p = end;
				//printf("%f ", f);
				linear_weights[count] = f;
				++count;
			}
		}
		else if(line_num == weights_rows + 3) {
			// biasParams
			char *p = line, *end;
			printf(" \n linear count : %d \n", count);
			count = 0;
			for (float f = strtof(p, &end); p != end; f = strtof(p, &end)) {
				p = end;
				//printf("%f ", f);
				bias_weights[count] = f;
				++count;
			}
			printf("bias count : %d", count);
		}

		++line_num;
	}
	
	printf("\nfirst linear weights: %f", linear_weights[0]);
	printf("\nfirst bias weights: %f", bias_weights[0]);

	fclose(fp);
}

