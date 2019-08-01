#ifndef TDNEURON_H_
#define TDNEURON_H_

typedef struct {
	float *weights;			// 连接的权重参数
	unsigned int nConnections;		// 连接数（权重数）
	float activation;		// 激活值
	float derivate;			// 激活函数的导数值
	float *inputs;			// 记录输入，用于计算梯度

}TDNeuron;

TDNeuron createTDNeuron(unsigned int nConnections);
float neuron_forward(TDNeuron* neuron, float* input);
float *backward(TDNeuron* neuron, float loss, float learningRate);

#endif /* TDNEURON_H_ */