#ifndef TDNET_H_
#define TDNET_H_

#include "TDLayer.h"
#include <stdio.h>

typedef struct {	
	TDLayer *layers;					// 层数据
	unsigned layersCount;				// 网络层数	
	unsigned inputDim;					// 一帧输入数据的维度
	int minOffset;						// 整个网络的最小偏移量
	int maxOffset;						// 整个网络的最大偏移量
	char* outputFilePath;
	char* midOutputFilePath;
} TDNet;

TDNet createTDNet(unsigned layersCount);
void addTDLayer(TDNet *net, const TDLayer *layer);
void forward(TDNet *net, float *input, FILE *fp);
void parseInputFile(const char*file, TDNet *net);
void computeBytes(TDNet *net);

#endif  /* TDNET_H_ */