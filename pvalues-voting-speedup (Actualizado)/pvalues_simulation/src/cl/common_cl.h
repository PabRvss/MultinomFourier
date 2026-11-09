#pragma once

//#define USE_DOUBLE_PRECISION

#ifdef USE_DOUBLE_PRECISION
typedef double real_t;
#else
typedef float real_t;
#endif

#define BIGM (1 << 30)
