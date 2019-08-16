#ifndef TDCONVLAYER_H_
#define TDCONVLAYER_H_

#include "TDUtils.h"

typedef struct {
	int time_offset;	
	int height_offset;
} Offset;

// ���ģ��
typedef struct {
	unsigned int num_filters_in;	// ���������������
	unsigned int in_h;				// ����ͼ��ĸ߶�

	unsigned int num_filters_out;	// ���������������
	unsigned int out_h;				// ���ͼ��ĸ߶�

	unsigned int stride_h;			// �߶ȷ���Ĳ�����height_subsample_out

	Offset* offsets;	
	unsigned int offset_nums;

	int* required_time_offsets;

} ConvModel;

typedef struct {
	float* w_params;
	float* b_params;
		
	TDShape* w_shape;
	//unsigned int w_param_w;  // weights width��num_filters_out
	//unsigned int w_param_h;  // weights height��num_filters_in * offset_nums
	//unsigned int w_param_c;  // weights channel

	unsigned int b_params_dim; // bias dim��num_filters_out
} ConvParam;

// �����
typedef struct {
	ConvModel *mode;
	ConvParam *param;
} TDConvLayer;

void conv_forward(TDConvLayer* layer, float* input, float* output);

#endif 

