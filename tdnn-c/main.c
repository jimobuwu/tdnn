#include <stdio.h>
#include "TDNet.h"
#include <stdlib.h>
#include <time.h>

int main() {
	struct TDNet net;
	srand(time(NULL));
	printf("%f", rand() / ((RAND_MAX + 1.0)));
	return 0;
}