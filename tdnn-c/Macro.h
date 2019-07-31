#pragma once

#define SAFEFREE(p) if(p){free(p);p=NULL;};
