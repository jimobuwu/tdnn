#ifndef TD_ACTIVATION_H
#define TD_ACTIVATION_H

#include <math.h>

float relu(float input) {
	return fmax(0.f, input);
}

float sigmoid(float input) {
	return 1.f / (1.f + expf(-input));
}

int logsoftmax(float *input, unsigned input_size) {
	if (!input)
		return -1;

	float max = input[0];
	float sum = 0.0f;
	for (int i = 0; i < input_size; ++i) {
		if (input[i] > max) {
			max = input[i];
		}
	}

	for (int i = 0; i < input_size; ++i) {
		sum += expf((input[i] -= max));
	}

	sum = logf(sum);
	
	for (int i = 0; i < input_size; ++i) {
		input[i] -= sum;
	}

	return 0;
}

#endif