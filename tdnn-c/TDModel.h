#ifndef TD_MODEL_H_
#define TD_MODEL_H_

typedef struct {
	char** componentNames;
} TDModel;

// base component
typedef struct {
	char *name;
	float *inputOffsets;
} TDComponent;

// input component
typedef struct {
	unsigned dim;
} TDInputComponent;

// bn
typedef struct {
	TDComponent *baseComponent;
	unsigned dim;
	unsigned blockDim;
	float epsilon;
	float *statsMean;
	float *statsVar;
} TDBNComponent;

// relu
typedef struct {
	TDComponent *baseComponent;
	unsigned dim;
	unsigned blockDim;
	float *valueAvg;	
} TDReluComponent;

// Dense
typedef struct {
	TDComponent *baseComponent;
	float learningRate;
	float *linearParams;
	float *biasParams;
} TDAffineComponent;

// Conv
typedef struct {
	TDComponent *baseComponent;
	float learningRate;
	unsigned numFiltersIn;
	unsigned numFiltersOut;
	unsigned heightIn;
	unsigned heightOut;
	unsigned heightSubsampleOut;
	float *offsets;
	float *requiredTimeOffsets;
	float *linearParams;
	float *biasParams;
} TDConvComponent;

#endif