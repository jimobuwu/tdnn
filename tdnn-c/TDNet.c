#include "TDNet.h"
#include <stdlib.h>
#include <stdio.h>
#include "Macro.h"
#include <string.h>
#include <omp.h>
#include "TDModel.h"

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
void forward(TDNet * net, float * input){
	//float *output = (float*)malloc(sizeof(float) * net->layers[net->layersCount - 1].neuronsCount);
	//if (!output) {
	//	fprintf(stderr, "train malloc output fail!");
	//	exit(1);
	//}
	
	float *passData = (float*)malloc(sizeof(float) * net->input_dim);
	if (!passData) {
		fprintf(stderr, "train malloc passData fail!");
		exit(1);
	}
	memcpy(passData, input, sizeof(float) * net->input_dim);

	// 前向
	for (int i = 0; i < net->layersCount; ++i) {
		TDLayer *layer = &net->layers[i];
		unsigned output_size = layer->neuronsCount * layer->neurons[0].height_out;
		float *activation = (float*)malloc(sizeof(float) * output_size);
		if (!activation) {
			fprintf(stderr, "in forward propagation, layer %d malloc activation fail!", i);
			exit(1);
		}

		layer_forward(layer, passData, activation);
		printf("\nlayer %d activations: \n", i);
		for (int m = 0; m < output_size; ++m) {
			if (0 == m % layer->neurons[0].height_out) {
				printf("\n");
			}
			printf("%f ", activation[m]);
		}

		// 最后一层输出
		//if (i == net->layersCount - 1)
		//	memcpy(output, activation, sizeof(float) * layer->neuronsCount);

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

//static parseModelLine(const char* line, TDNet *net, char *compNames[], int *compCount) {
//	char *pos;
//
//	if (strstr(line, "input-node")) {
//		
//	} 
//	else if(strstr(line, "component-node")) {
//		pos = strstr("line", "component");
//	}
//	else if (strstr(line, "output-node")) {
//	}
//	++compCount;
//}
//
//void parseModelFile(const char* file, TDNet *net) {
//	FILE *fp = fopen(file, "r");
//	if (!fp) {
//		return;
//	}
//
//	char *compNames[20];
//	int compCount = 0;
//
//	char line[LINE_BUF_SIZE];
//	while (!feof(fp)) {
//		fgets(line, LINE_BUF_SIZE, fp);
//		printf("%s", line);
//	}
//
//	fclose(fp);
//}

void parseInputFile(const char * file, TDNet *net) {
	FILE * fp = fopen(file, "r");
	if (!file) {
		return;
	}

	char line[LINE_BUF_SIZE];
	int line_num = 0;
	int count = 0;
	float *one_frame = (float*)calloc(net->input_dim, sizeof(float));

	while (!feof(fp)) {
		fgets(line, LINE_BUF_SIZE, fp);
		printf("\n line : ", line);

		count = 0;
		char *p = line, *end;
		for (float f = strtof(p, &end); p != end; f = strtof(p, &end)) {
			p = end;
			printf("%f ", f);
			one_frame[count] = f;
			++count;
		}	

		// 输入一帧数据，前向
		forward(net, one_frame);

		printf("\n");
		printf("\n one frame input count: %d", count);
	}
	printf("\n input count: %d", count);

	fclose(fp);
}