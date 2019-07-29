#ifndef TDNEURON_H_
#define TDNEURON_H_

typedef struct TDNeuron {
	float *weights;
	int nConnections;
	float activation;
	float derivate;

}TDNeuron;

TDNeuron createTDNeuron(int nConnections);
float forward(TDNeuron* neuron, float* input);
float *backward(TDNeuron* neuron);

#endif /* TDNEURON_H_ */