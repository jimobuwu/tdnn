#include "TDLayer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include "TDUtils.h"


TDLayer createTDLayer(unsigned int id, const char* name, LAYER_TYPE layer_type, 
	ACTIVATION_TYPE act_type, unsigned int neuronsCount, const TDShape *kernel_shape,
	const int *time_offsets, unsigned int offsets_size, const TDShape *input_shape, 
	unsigned int height_out, _Bool has_logsoftmax) {
	TDLayer layer;
	layer.id = id;
	layer.name = malloc(strlen(name) + 1);
	strcpy(layer.name, name);
	layer.type = layer_type;
	layer.neuronsCount = neuronsCount;
	layer.delay = kernel_shape->w;
	layer.curBufferFrameSize = 0;
	layer.has_logsoftmax = has_logsoftmax;
	layer.is_output = 0;
	
	layer.input_shape = (TDShape*)malloc(sizeof(TDShape));
	if (!layer.input_shape) {
		exit(1);
	}
	memcpy(layer.input_shape, input_shape, sizeof(TDShape));

	layer.neurons = (TDNeuron*)malloc(sizeof(TDNeuron) * neuronsCount);
	if (!layer.neurons) {
		exit(1);
	}

//#pragma omp parallel for
	for (int i = 0; i < neuronsCount; ++i) {
		layer.neurons[i] = createTDNeuron(act_type, kernel_shape, time_offsets, offsets_size, height_out);		
	}

	unsigned int inputFramesSize = (time_offsets[offsets_size - 1] - time_offsets[0] + 1) * layer.input_shape->c * layer.input_shape->h;
	layer.inputFramesSize = inputFramesSize;
	layer.inputFrames = (float*)calloc(inputFramesSize, sizeof(float));
	if (!layer.inputFrames) {
		exit(1);
	}

	return layer;
}

/*
return: 
-1, 延迟帧数不足
*/
static int layer_pushFrame(TDLayer *layer, float *input) {
	int ret = 0;
	unsigned int input_size = layer->input_shape->h * layer->input_shape->c;
	int offset = layer->neurons[0].time_offsets[layer->neurons[0].kernel_shape->w - 1];

	if (0 == layer->curBufferFrameSize) {
		for (unsigned int i = 0; i < layer->delay; ++i) {
			memcpy(&layer->inputFrames[i * input_size],
				input,
				sizeof(float) * input_size);
		}
	}

	//if (layer->curBufferFrameSize < offset + 1) {
	if (layer->curBufferFrameSize < layer->delay) {
		/*for (unsigned int i = 0; i < layer->delay - 1; ++i) {
			memcpy(&layer->inputFrames[i * input_size],
				&layer->inputFrames[(i + 1) * input_size],
				sizeof(float) * input_size);
		}
		memcpy(&layer->inputFrames[(layer->delay - 1) * input_size], input, sizeof(float) * input_size);*/

		// 加入新帧
		memcpy(&layer->inputFrames[layer->curBufferFrameSize * input_size], input, sizeof(float) * input_size);
		++layer->curBufferFrameSize;
		if(layer->curBufferFrameSize < layer->delay)
			ret = -1;
	} else {
		// 从第二帧开始前移
		for (unsigned int i = 0; i < layer->delay - 1; ++i) {
			memcpy(&layer->inputFrames[i * input_size],
				&layer->inputFrames[(i + 1) * input_size],
				sizeof(float) * input_size);
		}
		// 加入新帧
		memcpy(&layer->inputFrames[(layer->delay - 1) * input_size], input, sizeof(float) * input_size);
	}

	return ret;
}

// 根据timeoffsets下采样输入数据
static float* sampleInput(const TDNeuron *neuron, const float *input, const TDShape *input_shape) {
	unsigned one_input_size = input_shape->h * input_shape->c;

	float *sampled = (float*)malloc(sizeof(float) * neuron->kernel_shape->w * one_input_size);
	if (!sampled) {
		printf("malloc fail! sampleInput ");
		abort();
	}

	for (int i = 0; i < neuron->kernel_shape->w; ++i) {
		int offset = neuron->time_offsets[i] - neuron->time_offsets[0];
		memcpy(sampled + i * one_input_size, input + offset * one_input_size, sizeof(float) * one_input_size);
	}

	return sampled;
}

int layer_forward(TDLayer *layer, const float *input, float *output) {
	// 输入帧数不足，不计算
	if (-1 == layer_pushFrame(layer, input)) {
		return -1;
	}

	unsigned int height_out = layer->neurons[0].height_out;
	printf("\n\n input shape: { %d, %d, %d } \n", 1, layer->input_shape->h, layer->input_shape->c);
	printf("output shape: { %d, %d, %d }", 1, height_out, layer->neuronsCount);
	
	float *sampledInput = sampleInput(&layer->neurons[0], layer->inputFrames, layer->input_shape);

	for (unsigned int i = 0; i < layer->neuronsCount; ++i) {
		neuron_forward(&layer->neurons[i], sampledInput, layer->input_shape);
	}	

	for (unsigned int j = 0; j < height_out; ++j) {
		for (unsigned int i = 0; i < layer->neuronsCount; ++i) {
			output[j * layer->neuronsCount + i] = layer->neurons[i].activation[j];
		}
	}

	// logsoftmax
	if (layer->has_logsoftmax) {
		logsoftmax(output, layer->neuronsCount * height_out);
	}

	if (layer->is_output) {
		const char* output_file = "../../../data/hounet/output.txt";
		FILE *fp = fopen(output_file, "a+");
		//fprintf(fp, "layer id: %d \n", layer->id);
		int max_i = 0;
		int max_v = output[0];

		for (int i = 0; i < layer->neuronsCount * height_out; ++i) {
			if (max_v < output[i]) {
				max_v = output[i];
				max_i = i;
			}
			fprintf(fp, "%f ", output[i]);
		}
		fprintf(fp, "\nmax i = %d", max_i);
		fprintf(fp, "\n\n");
		fclose(fp);
	}

	return 0;
}

void load_weights(TDLayer * layer, const char *filePath) {
	TDShape * shape = layer->neurons[0].kernel_shape;
	unsigned flatten = shape->w * shape->h * shape->c;
	unsigned total = flatten * layer->neuronsCount;

	float *linear_weights = (float*)malloc(sizeof(float) * total);
	float *bias_weights = (float*)malloc(sizeof(float) * layer->neuronsCount);

	/*char *fileName = layer->name;
	size_t len = strlen(fileName);
	fileName = (char*)realloc(fileName, strlen(fileName) + 5);
	strcat(fileName, ".txt");*/

	parseWeights(filePath, layer->neuronsCount, linear_weights, bias_weights);

	unsigned index = 0;
	for (int i = 0; i < layer->neuronsCount; ++i, index = i * flatten) {
		memcpy(layer->neurons[i].weights, linear_weights + index, sizeof(float) * flatten);
		layer->neurons[i].bias = bias_weights[i];
	}
}

void addBN(TDLayer *layer, const char* filePath, unsigned dim, float epsilon, float gamma) {
	// load from file
	FILE *fp = fopen(filePath, "r");
	if (!fp) {
		return;
	}

	float *means = (float*)malloc(sizeof(float) * dim);
	float *vars = (float*)malloc(sizeof(float) * dim);

	char line[LINE_BUF_SIZE];
	int lineNum = 0;
	int weights_count = 0;

	while (!feof(fp)) {
		fgets(line, LINE_BUF_SIZE, fp);

		if (1 == lineNum) {
			printf("\naddBN mean:\n");
			char *p = line, *end;
			for (float f = strtof(p, &end); p != end; f = strtof(p, &end)) {
				p = end;
				means[weights_count] = f;
				printf("%f ", f);
				++weights_count;
			}
		}
		else if (4 == lineNum) {
			weights_count = 0;
			printf("\naddBN vars:\n");
			char *p = line, *end;
			for (float f = strtof(p, &end); p != end; f = strtof(p, &end)) {
				p = end;
				vars[weights_count] = f;
				printf("%f ", f);
				++weights_count;
			}
		}
		
		++lineNum;
	}

	for (int i = 0; i < layer->neuronsCount; ++i) {
		addNeuronBN(&layer->neurons[i], epsilon, gamma, means[i], vars[i]);
	}
}


