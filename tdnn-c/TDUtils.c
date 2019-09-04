#include "TDUtils.h"
#include "assert.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

float* getConv(const float *input, const TDShape *inputShape,
	const float *kernel, const TDShape *kernelShape, unsigned int stride_h) {

	int out_w = 1;
	int out_h = (inputShape->h - kernelShape->h) / stride_h + 1;
		
	const int conv_len = kernelShape->w * kernelShape->h * kernelShape->c;
	float* tmp = (float*)calloc(out_w * out_h * conv_len, sizeof(float));
	if (!tmp) {
		abort();
	}

	int input_w = inputShape->h * inputShape->c;
	int index = 0;

	// 权重按照 w,h,c 排列
	for (int i = 0; i < out_h; ++i) {
		for (int w = 0; w < kernelShape->w; ++w) {
			for (int h = 0; h < kernelShape->h * kernelShape->c; ++h) {
				int start = i * kernelShape->c + input_w * w;
				tmp[index++] = input[start + h];
			}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
		}
	}

	float* out = (float*)malloc(sizeof(float) * out_w * out_h);
	if (!out) {
		printf("malloc fail! getConv out ");
		abort();
	}

	int wh = 0;
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

		if (line_num >= 1 && line_num <= weights_rows + 1) {
			// linearParams
			char *p = line, *end;			
			for (float f = strtof(p, &end); p != end; f = strtof(p, &end)) {
				p = end;
				linear_weights[count] = f;
				++count;
			}
		}
		else if(line_num == weights_rows + 3) {
			// biasParams
			char *p = line, *end;
			//printf(" \n linear count : %d \n", count);
			count = 0;
			for (float f = strtof(p, &end); p != end; f = strtof(p, &end)) {
				p = end;
				bias_weights[count] = f;
				++count;
			}
			//printf("bias count : %d", count);
		}

		++line_num;
	}
	
	//printf("\nfirst linear weights: %f", linear_weights[0]);
	//printf("\nfirst bias weights: %f", bias_weights[0]);

	fclose(fp);
}

