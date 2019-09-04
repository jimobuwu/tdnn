#include "TDNet.h"
#include <stdlib.h>
#include "Macro.h"
#include <string.h>
#include <omp.h>
#include "TDModel.h"

int frame_input_num = 0;

TDNet createTDNet(unsigned layersCount) {
	TDNet net;
	net.layersCount = layersCount;
	net.layers = (TDLayer*)malloc(sizeof(TDLayer) * layersCount);

	return net;
}

void addTDLayer(TDNet *net, const TDLayer *layer) {
	memcpy(&net->layers[layer->id], layer, sizeof(TDLayer));
}

// input, 一帧数据
void forward(TDNet * net, float * input, FILE *fp){
	++frame_input_num;
	printf("\n input frame num: %d", frame_input_num);

	float *passData = (float*)malloc(sizeof(float) * net->input_dim);
	if (!passData) {
		fprintf(stderr, "train malloc passData fail!");
		exit(1);
	}
	memcpy(passData, input, sizeof(float) * net->input_dim);

	// 前向
	for (int i = 0; i < net->layersCount; ++i) {
		TDLayer *layer = &net->layers[i];
		unsigned output_size = layer->neuronsCount * layer->neurons[0].heightOut;
		float *activation = (float*)malloc(sizeof(float) * output_size);
		if (!activation) {
			fprintf(stderr, "in forward propagation, layer %d malloc activation fail!", i);
			exit(1);
		}

		if( -1 == layer_forward(layer, passData, activation, net->outputFilePath)) {
			// 延迟帧数不足时，继续接收输入数据
			break;
		}              

		fprintf(fp, "\nlayer %d activations: \n", i);
		for (int m = 0; m < output_size; ++m) {
			/*if (0 == m % layer->neurons[0].heightOut) {
				fprintf(fp, "\n");
			}*/
			fprintf(fp, "%f ", activation[m]);
		}
		fprintf(fp, "\n\n");

		// 上一层的激活值作为下一层的输入
		passData = (float*)realloc(passData, sizeof(float) * output_size);
		if (!passData) {
			fprintf(stderr, "in forward propagation, layer %d realloc passData fail!", i);
			exit(1);
		}

		memcpy(passData, activation, sizeof(float) * output_size);
		SAFEFREE(activation);
	}


	SAFEFREE(passData);
}

void parseInputFile(const char *file, TDNet *net) {
	FILE * fp = fopen(file, "r");
	if (!fp) {
		return;
	}

	char line[LINE_BUF_SIZE];
	int line_num = 0;
	int count = 0;
	float *one_frame = (float*)calloc(net->input_dim, sizeof(float));

	FILE *ouput_fp = fopen(net->midOutputFilePath, "w");
	if (!ouput_fp) {
		return;
	}

	while (!feof(fp)) {
		fgets(line, LINE_BUF_SIZE, fp);
		printf("\n input line : ", line);
		printf("\n");

		count = 0;
		char *p = line, *end;
		for (float f = strtof(p, &end); p != end; f = strtof(p, &end)) {
			p = end;
			printf("%f ", f);
			one_frame[count] = f;
			++count;
		}	

		// 输入一帧数据，前向
		forward(net, one_frame, ouput_fp);
		//printf("\n one frame input count: %d", count);
	}
	//printf("\n input count: %d", count);

	fclose(fp);
	fclose(ouput_fp);
}