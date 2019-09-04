#ifndef MACRO_H_
#define MACRO_H_

#define SAFEFREE(p) if(p){free(p);p=NULL;};

#define LINE_BUF_SIZE 25360 

typedef enum {
	DENSE,
	CONV
}layerType;

typedef enum {
	SAME,
	VALID
}PADDING_TYPE;

typedef enum {
	NONE_ACT,
	RELU,
	SIGMOID,
	SOFTMAX
} ACTIVATION_TYPE;

typedef enum {
	NONE_NORM,
	BN,
	NORM	
} NORM_TYPE;


#endif
