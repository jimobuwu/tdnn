#ifndef TDNEURON_H_
#define TDNEURON_H_

typedef struct {
	float *weights;					// 连接的权重参数
	unsigned int kernel_w;			// 卷积核宽度
	unsigned int kernel_h;			// 卷积核高度
	unsigned int nConnections;		// 连接数（权重数）
	float *activation;				// 激活值	
	float *inputs;					// 记录输入，用于计算梯度

}TDNeuron;

TDNeuron createTDNeuron(unsigned int nConnections, unsigned int kernel_w, unsigned int kernel_h);
float *neuron_forward(TDNeuron* neuron, float* input, unsigned int input_w, unsigned int input_h);
float *backward(TDNeuron* neuron, float loss, float learningRate);

#endif /* TDNEURON_H_ */