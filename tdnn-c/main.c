#include <stdio.h>
#include "TDNet.h"
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>
#include <omp.h>
#include <python3.6/Python.h>

// hounet
static TDNet createHouNet() {
	TDNet net = createTDNet(7);
	net.inputDim = 78;

	// conv1
	TDShape kernelShape1 = { 8, 18, 3 };
	TDShape inputShape1 = {1, 26, 3};
	int timeOffsets1[8] = { -4, -3, -2, -1, 0, 1, 2, 3 };

	TDLayer l1 = createTDLayer(0, "cnn1.conv", CONV, RELU, 128, &kernelShape1, timeOffsets1, 8, &inputShape1, 9, 0, 0);
	load_weights(&l1, "../../../data/hounet/cnn1.conv.txt");
	addBN(&l1, "../../../data/hounet/cnn1.batchnorm.txt", 128, 0.001, 1.f);
	addTDLayer(&net, &l1);

	//conv2
	TDShape kernelShape2 = { 4, 3, 128 };
	TDShape inputShape2 = { 1, 9, 128 };
	int timeOffsets2[4] = { -2, -1, 0, 1};

	TDLayer l2 = createTDLayer(1, "cnn2.conv", CONV, RELU, 64, &kernelShape2, timeOffsets2, 4, &inputShape2, 7, 0, 0);
	load_weights(&l2, "../../../data/hounet/cnn2.conv.txt");
	addBN(&l2, "../../../data/hounet/cnn2.batchnorm.txt", 64, 0.001, 1.f);
	addTDLayer(&net, &l2);

	////conv3
	TDShape kernelShape3 = { 4, 3, 64 };
	TDShape inputShape3 = { 1, 7, 64 };
	int timeOffsets3[4] = { -2, -1, 0, 1 };
	TDLayer l3 = createTDLayer(2, "cnn3.conv", CONV, RELU, 64, &kernelShape3, timeOffsets3, 4, &inputShape3, 5, 0, 0);
	load_weights(&l3, "../../../data/hounet/cnn3.conv.txt");
	addBN(&l3, "../../../data/hounet/cnn3.batchnorm.txt", 64, 0.001, 1.f);
	addTDLayer(&net, &l3);
	
	// Affine1
	TDShape kernelShape4 = { 3, 5, 64 };
	TDShape inputShape4 = { 1, 5, 64 };
	int timeOffsets4[3] = { -1, 0, 1 };
	TDLayer l4 = createTDLayer(3, "Affine1", DENSE, RELU, 512, &kernelShape4, timeOffsets4, 3, &inputShape4, 1, 0, 0);
	load_weights(&l4, "../../../data/hounet/Affine1.txt");
	addTDLayer(&net, &l4);

	// Affine2
	TDShape kernelShape5 = { 1, 512, 1 };
	TDShape inputShape5 = { 1, 512, 1 };
	int timeOffsets5[1] = { 0 };
	TDLayer l5 = createTDLayer(4, "Affine2", DENSE, RELU, 512, &kernelShape5, timeOffsets5, 1, &inputShape5, 1, 0, 0);
	load_weights(&l5, "../../../data/hounet/Affine2.txt");
	addTDLayer(&net, &l5);

	// Affine3
	TDShape kernelShape6 = { 1, 512, 1 };
	TDShape inputShape6 = { 1, 512, 1 };
	int timeOffsets6[1] = { 0 };
	TDLayer l6 = createTDLayer(5, "Affine3", DENSE, RELU, 512, &kernelShape6, timeOffsets6, 1, &inputShape6, 1, 0, 0);
	load_weights(&l6, "../../../data/hounet/Affine3.txt");
	addTDLayer(&net, &l6);

	// final Affine
	TDLayer l7 = createTDLayer(6, "Final_affine", DENSE, NONE_ACT, 67, &kernelShape6, timeOffsets6, 1, &inputShape6, 1, 1, 0);
	load_weights(&l7, "../../../data/hounet/Final_affine.txt");
	l7.isOutput = 1;
	addTDLayer(&net, &l7);

	return net;
}

// 11000net
static TDNet create11000Net() {
	TDNet net = createTDNet(8);
	net.inputDim = 50;

	// dense 0
	TDShape kernelShape0 = { 5, 50, 1 };
	TDShape inputShape0 = { 1, 50, 1 };
	int timeOffsets0[5] = { -2, -1, 0, 1, 2 };

	TDLayer l0 = createTDLayer(0, "dense0", DENSE, NONE_ACT, 250, &kernelShape0, timeOffsets0, 5, &inputShape0, 1, 0, 1);
	load_weights(&l0, "../../../data/11000net/dense0.txt");
	//l0.isOutput = 1;
	addTDLayer(&net, &l0);

	// dense 1
	TDShape kernelShape1 = { 1, 250, 1 };
	TDShape inputShape1 = { 1, 250, 1 };
	int timeOffsets1[1] = { 0 };

	TDLayer l1 = createTDLayer(1, "dense1", DENSE, RELU, 256, &kernelShape1, timeOffsets1, 1, &inputShape1, 1, 0, 1);
	load_weights(&l1, "../../../data/11000net/dense1.txt");
	addLayerNorm(&l1, 1.f);
	addTDLayer(&net, &l1);

	// dense 2
	TDShape kernelShape2 = { 2, 256, 1 };
	TDShape inputShape2 = { 1, 256, 1 };
	int timeOffsets2[2] = { -1, 2 };

	TDLayer l2 = createTDLayer(2, "dense2", DENSE, RELU, 256, &kernelShape2, timeOffsets2, 2, &inputShape2, 1, 0, 1);
	load_weights(&l2, "../../../data/11000net/dense2.txt");
	addLayerNorm(&l2, 1.f);
	addTDLayer(&net, &l2);

	// dense 3
	int timeOffsets3[2] = { -2, 1 };
	TDLayer l3 = createTDLayer(3, "dense3", DENSE, RELU, 256, &kernelShape2, timeOffsets3, 2, &inputShape2, 1, 0, 1);
	load_weights(&l3, "../../../data/11000net/dense3.txt");
	addLayerNorm(&l3, 1.f);
	addTDLayer(&net, &l3);

	// dense 4
	int timeOffsets4[2] = { -3, 3 };
	TDLayer l4 = createTDLayer(4, "dense4", DENSE, RELU, 256, &kernelShape2, timeOffsets4, 2, &inputShape2, 1, 0, 1);
	load_weights(&l4, "../../../data/11000net/dense4.txt");
	addLayerNorm(&l4, 1.f);
	addTDLayer(&net, &l4);

	// dense 5
	int timeOffsets5[2] = { -5, 1 };
	TDLayer l5 = createTDLayer(5, "dense5", DENSE, RELU, 256, &kernelShape2, timeOffsets5, 2, &inputShape2, 1, 0, 1);
	load_weights(&l5, "../../../data/11000net/dense5.txt");
	addLayerNorm(&l5, 1.f);
	addTDLayer(&net, &l5);

	// dense 6
	TDShape kernelShape6 = { 1, 256, 1 };
	TDShape inputShape6 = { 1, 256, 1 };
	int timeOffsets6[1] = { 0 };
	TDLayer l6 = createTDLayer(6, "dense6", DENSE, RELU, 256, &kernelShape6, timeOffsets6, 1, &inputShape6, 1, 0, 1);
	load_weights(&l6, "../../../data/11000net/dense6.txt");
	addLayerNorm(&l6, 0.5f);
	addTDLayer(&net, &l6);

	// dense 7
	TDLayer l7 = createTDLayer(7, "dense7", DENSE, NONE_ACT, 889, &kernelShape6, timeOffsets6, 1, &inputShape6, 1, 0, 1);
	load_weights(&l7, "../../../data/11000net/dense7.txt");
	l7.isOutput = 1;
	addTDLayer(&net, &l7);

	return net;
}

// kwsnet
static TDNet createKwsNet() {
	TDNet net = createTDNet(10);
	net.inputDim = 40;

	// dense 0
	TDShape kernelShape0 = { 5, 40, 1 };
	TDShape inputShape0 = { 1, 40, 1 };
	int timeOffsets0[5] = { -2, -1, 0, 1, 2 };

	TDLayer l0 = createTDLayer(0, "dense0", DENSE, NONE_ACT, 200, &kernelShape0, timeOffsets0, 5, &inputShape0, 1, 0, 1);
	load_weights(&l0, "../../../data/kws/dense0.txt");	
	addTDLayer(&net, &l0);

	// dense 1
	TDShape kernelShape1 = { 1, 200, 1 };
	TDShape inputShape1 = { 1, 200, 1 };
	int timeOffsets1[1] = { 0 };

	TDLayer l1 = createTDLayer(1, "dense1", DENSE, RELU, 100, &kernelShape1, timeOffsets1, 1, &inputShape1, 1, 0, 1);
	load_weights(&l1, "../../../data/kws/dense1.txt");
	addTDLayer(&net, &l1);

	// dense 2
	TDShape kernelShape2 = { 1, 100, 1 };
	TDShape inputShape2 = { 1, 100, 1 };
	int timeOffsets2[1] = { 0 };

	TDLayer l2 = createTDLayer(2, "dense2", DENSE, NONE_ACT, 55, &kernelShape2, timeOffsets2, 1, &inputShape2, 1, 0, 1);
	load_weights(&l2, "../../../data/kws/dense2.txt");
	addTDLayer(&net, &l2);

	// dense 3
	TDShape kernelShape3 = { 2, 55, 1 };
	TDShape inputShape3 = { 1, 55, 1 };
	int timeOffsets3[2] = { -2, 2 };

	TDLayer l3 = createTDLayer(3, "dense3", DENSE, RELU, 386, &kernelShape3, timeOffsets3, 2, &inputShape3, 1, 0, 1);
	load_weights(&l3, "../../../data/kws/dense3.txt");
	addTDLayer(&net, &l3);

	// dense 3-1
	TDShape kernelShape3_1 = { 1, 386, 1 };
	TDShape inputShape3_1 = { 1, 386, 1 };
	int timeOffsets3_1[1] = { 0 };

	TDLayer l3_1 = createTDLayer(4, "dense3_1", DENSE, NONE_ACT, 55, &kernelShape3_1, timeOffsets1, 1, &inputShape3_1, 1, 0, 1);
	load_weights(&l3_1, "../../../data/kws/dense3_1.txt");
	addTDLayer(&net, &l3_1);

	// dense 4
	TDShape kernelShape4 = { 2, 55, 1 };
	TDShape inputShape4 = { 1, 55, 1 };
	int timeOffsets4[2] = { -4, 4 };
	TDLayer l4 = createTDLayer(5, "dense4", DENSE, RELU, 386, &kernelShape4, timeOffsets4, 2, &inputShape4, 1, 0, 1);
	load_weights(&l4, "../../../data/kws/dense4.txt");
	addTDLayer(&net, &l4);

	// dense 4-1
	TDShape kernelShape4_1 = { 1, 386, 1 };
	TDShape inputShape4_1 = { 1, 386, 1 };
	int timeOffsets4_1[1] = { 0 };

	TDLayer l4_1 = createTDLayer(6, "dense4_1", DENSE, NONE_ACT, 55, &kernelShape4_1, timeOffsets1, 1, &inputShape4_1, 1, 0, 1);
	load_weights(&l4_1, "../../../data/kws/dense4_1.txt");
	addTDLayer(&net, &l4_1);

	// dense 5
	TDShape kernelShape5 = { 2, 55, 1 };
	TDShape inputShape5 = { 1, 55, 1 };
	int timeOffsets5[2] = { -12, 2 };
	TDLayer l5 = createTDLayer(7, "dense5", DENSE, RELU, 386, &kernelShape5, timeOffsets5, 2, &inputShape5, 1, 0, 1);
	load_weights(&l5, "../../../data/kws/dense5.txt");	
	addTDLayer(&net, &l5);

	// dense 6
	TDShape kernelShape6 = { 1, 386, 1 };
	TDShape inputShape6 = { 1, 386, 1 };
	int timeOffsets6[1] = { 0 };
	TDLayer l6 = createTDLayer(8, "dense6", DENSE, RELU, 386, &kernelShape6, timeOffsets6, 1, &inputShape6, 1, 0, 1);
	load_weights(&l6, "../../../data/kws/dense6.txt");
	addLayerNorm(&l6, 0.5f);
	addTDLayer(&net, &l6);

	// dense 7
	TDLayer l7 = createTDLayer(9, "dense7", DENSE, NONE_ACT, 81, &kernelShape6, timeOffsets6, 1, &inputShape6, 1, 0, 1);
	load_weights(&l7, "../../../data/kws/dense7.txt");
	l7.isOutput = 1;
	addTDLayer(&net, &l7);

	return net;
}

int main() {
	//TDNet net = createHouNet();	
	//net.outputFilePath = "../../../data/hounet/output.txt";
	//net.midOutputFilePath = "../../../data/hounet/mid-output.txt";
	//parseInputFile("../../../data/hounet/input.txt", &net);
	
	//TDNet net = create11000Net();	
	//net.outputFilePath = "../../../data/11000net/output.txt";
	//net.midOutputFilePath = "../../../data/11000net/mid-output.txt";
	//parseInputFile("../../../data/11000net/feats_01.txt", &net);

	/*Py_Initialize();
	if (!Py_isInitialized()) {
		printf("python init failed!\n");
		return -1;
	}

	PyRun_SimpleString("import numpy as np");*/


	TDNet net = createKwsNet();
	net.outputFilePath = "../../../data/kws/output.txt";
	net.midOutputFilePath = "../../../data/kws/mid-output.txt";
	parseInputFile("../../../data/kws/input_kws.txt", &net); 

	// computeBytes(&net);

	//Py_Finalize();

	return 0;
}