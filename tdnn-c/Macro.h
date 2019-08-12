#ifndef MACRO_H_
#define MACRO_H_

#define SAFEFREE(p) if(p){free(p);p=NULL;};

typedef enum{
	DENSE,
	CONV
}LAYER_TYPE;

typedef enum {
	SAME,
	VALID
}PADDING_TYPE;

#endif
