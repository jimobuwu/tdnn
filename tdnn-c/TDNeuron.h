#ifndef TDNEURON_H_
#define TDNEURON_H_

typedef struct {
	float *weights;			// ���ӵ�Ȩ�ز���
	unsigned int nConnections;		// ��������Ȩ������
	float activation;		// ����ֵ
	float derivate;			// ������ĵ���ֵ
	float *inputs;			// ��¼���룬���ڼ����ݶ�

}TDNeuron;

TDNeuron createTDNeuron(unsigned int nConnections);
float neuron_forward(TDNeuron* neuron, float* input);
float *backward(TDNeuron* neuron, float loss, float learningRate);

#endif /* TDNEURON_H_ */