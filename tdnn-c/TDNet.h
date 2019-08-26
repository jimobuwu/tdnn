#ifndef TDNET_H_
#define TDNET_H_

#include "TDLayer.h"
#include <stdio.h>

typedef struct {	
	TDLayer *layers;					// ������
	unsigned layersCount;			// �������	
	unsigned input_dim;					// һ֡�������ݵ�ά��
} TDNet;

TDNet createTDNet(unsigned layersCount);
void addTDLayer(TDNet *net, const TDLayer *layer);
void forward(TDNet *net, float *input, FILE *fp);
void parseInputFile(const char*file, TDNet *net);

#endif  /* TDNET_H_ */