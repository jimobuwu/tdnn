#include "TDLayer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
#include "TDUtils.h"


TDLayer createTDLayer(unsigned int id, const char *name, layerType layerType, 
	ACTIVATION_TYPE actType, unsigned int neuronsCount, const TDShape *kernelShape,
	const int *timeOffsets, unsigned int offsetsSize, const TDShape *inputShape, 
	unsigned int heightOut, _Bool hasLogsoftmax, _Bool actBeforeNorm) {
	TDLayer layer;
	layer.id = id;
	layer.name = malloc(strlen(name) + 1);
	strcpy(layer.name, name);
	layer.type = layerType;
	layer.neuronsCount = neuronsCount;
	layer.delay = timeOffsets[offsetsSize - 1] - timeOffsets[0] + 1;
	layer.curBufferFrameSize = 0;
	layer.hasLogsoftmax = hasLogsoftmax;
	layer.isOutput = 0;
	layer.LNTargetRms = 1.f;
	
	layer.inputShape = (TDShape*)malloc(sizeof(TDShape));
	if (!layer.inputShape) {
		exit(1);
	}
	memcpy(layer.inputShape, inputShape, sizeof(TDShape));

	layer.neurons = (TDNeuron*)malloc(sizeof(TDNeuron) * neuronsCount);
	if (!layer.neurons) {
		exit(1);
	}

	for (int i = 0; i < neuronsCount; ++i) {
		layer.neurons[i] = createTDNeuron(actType, kernelShape, timeOffsets, offsetsSize, heightOut, actBeforeNorm);
	}

	unsigned int inputFramesSize = layer.delay * layer.inputShape->c * layer.inputShape->h;
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
	unsigned int input_size = layer->inputShape->h * layer->inputShape->c;

	if (layer->curBufferFrameSize < layer->delay) {
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
static float* sampleInput(const TDNeuron *neuron, const float *input, const TDShape *inputShape) {
	unsigned one_input_size = inputShape->h * inputShape->c;

	float *sampled = (float*)malloc(sizeof(float) * neuron->kernelShape->w * one_input_size);
	if (!sampled) {
		printf("malloc fail! sampleInput ");
		abort();
	}

	for (int i = 0; i < neuron->kernelShape->w; ++i) {
		int offset = neuron->timeOffsets[i] - neuron->timeOffsets[0];
		memcpy(sampled + i * one_input_size, input + offset * one_input_size, sizeof(float) * one_input_size);
	}

	return sampled;
}

int layer_forward(TDLayer *layer, const float *input, float *output, const char *outputFilePath) {
	// 输入帧数不足，不计算
	if (-1 == layer_pushFrame(layer, input)) {
		return -1;
	}

	unsigned int heightOut = layer->neurons[0].heightOut;
	float *sampledInput = sampleInput(&layer->neurons[0], layer->inputFrames, layer->inputShape);

	TDShape *kShape = layer->neurons[0].kernelShape;
	int out_w = 1;
	int out_h = (layer->inputShape->h - kShape->h) / layer->neurons[0].stride_h + 1;
	const int conv_len = kShape->w * kShape->h * kShape->c;

	for (unsigned int i = 0; i < layer->neuronsCount; ++i) {
		neuron_forward(&layer->neurons[i], sampledInput, layer->inputShape);
	}	

	float sum = 0.f;
	for (unsigned int j = 0; j < heightOut; ++j) {
		for (unsigned int i = 0; i < layer->neuronsCount; ++i) {
			output[j * layer->neuronsCount + i] = layer->neurons[i].activation[j];
			sum += output[j * layer->neuronsCount + i] * output[j * layer->neuronsCount + i];
		}
	}
	
	if (layer->hasLN) {
		float scale = powf(fmax(0.f, sum / (layer->neuronsCount * heightOut * layer->LNTargetRms * layer->LNTargetRms)), -0.5);
		for (unsigned int j = 0; j < heightOut; ++j) {
			for (unsigned int i = 0; i < layer->neuronsCount; ++i) {
				output[j * layer->neuronsCount + i] = scale * output[j * layer->neuronsCount + i];
			}
		}
	}

	// logsoftmax
	if (layer->hasLogsoftmax) {
		logsoftmax(output, layer->neuronsCount * heightOut);
	}

	// write output to file
	if (layer->isOutput) {
		FILE *fp = fopen(outputFilePath, "a+");
		int max_i = 0;
		int max_v = output[0];

		for (int i = 0; i < layer->neuronsCount * heightOut; ++i) {
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

void load_weights(TDLayer *layer, const char *filePath) {
	TDShape * shape = layer->neurons[0].kernelShape;
	unsigned flatten = shape->w * shape->h * shape->c;
	unsigned total = flatten * layer->neuronsCount;

	float *linear_weights = (float*)malloc(sizeof(float) * total);
	float *bias_weights = (float*)calloc(layer->neuronsCount, sizeof(float));
	printf("layer %d load weights tmp buffer bytes: %d\n", layer->id, total * sizeof(float));

	parseWeights(filePath, layer->neuronsCount, linear_weights, bias_weights);

	unsigned index = 0;
	for (int i = 0; i < layer->neuronsCount; ++i, index = i * flatten) {
		memcpy(layer->neurons[i].weights, linear_weights + index, sizeof(float) * flatten);
		layer->neurons[i].bias = bias_weights[i];
	}

	SAFEFREE(linear_weights);
	SAFEFREE(bias_weights);
}

void addBN(TDLayer *layer, const char *filePath, unsigned dim, float epsilon, float gamma) {
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
			//printf("\naddBN mean:\n");
			char *p = line, *end;
			for (float f = strtof(p, &end); p != end; f = strtof(p, &end)) {
				p = end;
				means[weights_count] = f;
				//printf("%f ", f);
				++weights_count;
			}
		}
		else if (4 == lineNum) {
			weights_count = 0;
			//printf("\naddBN vars:\n");
			char *p = line, *end;
			for (float f = strtof(p, &end); p != end; f = strtof(p, &end)) {
				p = end;
				vars[weights_count] = f;
				//printf("%f ", f);
				++weights_count;
			}
		}
		
		++lineNum;
	}

	for (int i = 0; i < layer->neuronsCount; ++i) {
		addNeuronBN(&layer->neurons[i], epsilon, gamma, means[i], vars[i]);
	}

	SAFEFREE(means);
	SAFEFREE(vars);
}

void addLayerNorm(TDLayer *layer, float targetRms) {
	layer->hasLN = 1;
	layer->LNTargetRms = targetRms;
}


