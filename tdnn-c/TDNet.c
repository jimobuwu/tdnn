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

// input, һ֡����
void forward(TDNet * net, float * input, FILE *fp){
	//float *output = (float*)malloc(sizeof(float) * net->layers[net->layersCount - 1].neuronsCount);
	//if (!output) {
	//	fprintf(stderr, "train malloc output fail!");
	//	exit(1);
	//}
	
	++frame_input_num;
	printf("\n input frame num: %d", frame_input_num);

	float *passData = (float*)malloc(sizeof(float) * net->input_dim);
	if (!passData) {
		fprintf(stderr, "train malloc passData fail!");
		exit(1);
	}
	memcpy(passData, input, sizeof(float) * net->input_dim);

	// ǰ��
	for (int i = 0; i < net->layersCount; ++i) {
		TDLayer *layer = &net->layers[i];
		unsigned output_size = layer->neuronsCount * layer->neurons[0].height_out;
		float *activation = (float*)malloc(sizeof(float) * output_size);
		if (!activation) {
			fprintf(stderr, "in forward propagation, layer %d malloc activation fail!", i);
			exit(1);
		}

		if( -1 == layer_forward(layer, passData, activation)) {
			// �ӳ�֡������ʱ������������������
			break;
		}

		fprintf(fp, "\nlayer %d activations: \n", i);
		for (int m = 0; m < output_size; ++m) {
			if (0 == m % layer->neurons[0].height_out) {
				fprintf(fp, "\n");
			}

			fprintf(fp, "%f ", activation[m]);
		}
		fprintf(fp, "\n\n");

		/*printf("\nlayer %d activations: \n", i);
		for (int m = 0; m < output_size; ++m) {
			if (0 == m % layer->neurons[0].height_out) {
				printf("\n");
			}
			printf("%f ", activation[m]);
		}*/

		// ��һ��ļ���ֵ��Ϊ��һ�������
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

static void afterHandle(TDNet *net) {
	// ���һ֡��������
	for (int i = 0; i < net->layersCount; ++i) {
		TDLayer *layer = &net->layers[i];
		int offset = layer->neurons[0].time_offsets[layer->neurons[0].kernel_shape->w - 1];
		unsigned inputSize = sizeof(float) * layer->input_shape->c * layer->input_shape->h;
		float *lastFrame = (float*)malloc(inputSize);

		if (layer->curBufferFrameSize == offset) {
			memcpy(lastFrame, &layer->inputFrames[layer->delay - 1], inputSize);
			
		}
		else { /* layer->curBufferFrameSize < offset, ֡������*/
			memcpy(lastFrame, &layer->inputFrames[layer->curBufferFrameSize - 1], inputSize);
		}
	}
}

void parseInputFile(const char * file, TDNet *net) {
	FILE * fp = fopen(file, "r");
	if (!fp) {
		return;
	}

	char line[LINE_BUF_SIZE];
	int line_num = 0;
	int count = 0;
	float *one_frame = (float*)calloc(net->input_dim, sizeof(float));

	const char* output_file = "../../../data/hounet/mid-output.txt";
	FILE *ouput_fp = fopen(output_file, "w");

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

		// ����һ֡���ݣ�ǰ��
		forward(net, one_frame, ouput_fp);
		printf("\n one frame input count: %d", count);
	}
	printf("\n input count: %d", count);

	// ������offset�����һ֡
	/*int offset = net->layers[0].neurons[0].time_offsets[net->layers[0].neurons[0].kernel_shape->w - 1];
	for (int i = 0; i < offset; ++i) {
		forward(net, one_frame, ouput_fp);
	}*/

	fclose(fp);
	fclose(ouput_fp);
}