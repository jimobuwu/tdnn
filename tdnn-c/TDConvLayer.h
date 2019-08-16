#ifndef TDCONVLAYER_H_
#define TDCONVLAYER_H_

#include "TDUtils.h"

typedef struct {
	int time_offset;	
	int height_offset;
} Offset;

// 卷积模型
typedef struct {
	unsigned int num_filters_in;	// 输入过滤器的数量
	unsigned int in_h;				// 输入图像的高度

	unsigned int num_filters_out;	// 输出过滤器的数量
	unsigned int out_h;				// 输出图像的高度

	unsigned int stride_h;			// 高度方向的步长，height_subsample_out

	Offset* offsets;	
	unsigned int offset_nums;

	int* required_time_offsets;

} ConvModel;

typedef struct {
	float* w_params;
	float* b_params;
		
	TDShape* w_shape;
	//unsigned int w_param_w;  // weights width，num_filters_out
	//unsigned int w_param_h;  // weights height，num_filters_in * offset_nums
	//unsigned int w_param_c;  // weights channel

	unsigned int b_params_dim; // bias dim，num_filters_out
} ConvParam;

// 卷积层
typedef struct {
	ConvModel *mode;
	ConvParam *param;
} TDConvLayer;

void conv_forward(TDConvLayer* layer, float* input, float* output);

#endif 

