#include <stdio.h>
#include "TDNet.h"
#include <stdlib.h>
#include <time.h>

int main() {
	//struct TDNet net;
	srand(time(NULL));
	//printf("%f", rand() / ((RAND_MAX + 1.0)));

	float *passData = (float*)malloc(sizeof(float) * 5);
	printf("%d", sizeof(float) * 5);

	return 0;
}