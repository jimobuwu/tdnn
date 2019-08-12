#ifndef TDNEURON_H_
#define TDNEURON_H_

typedef struct {
	float *weights;					// ���ӵ�Ȩ�ز���
	unsigned int kernel_w;			// ����˿��
	unsigned int kernel_h;			// ����˸߶�
	unsigned int nConnections;		// ��������Ȩ������
	float *activation;				// ����ֵ	
	float *inputs;					// ��¼���룬���ڼ����ݶ�

}TDNeuron;

TDNeuron createTDNeuron(unsigned int nConnections, unsigned int kernel_w, unsigned int kernel_h);
float *neuron_forward(TDNeuron* neuron, float* input, unsigned int input_w, unsigned int input_h);
float *backward(TDNeuron* neuron, float loss, float learningRate);

#endif /* TDNEURON_H_ */