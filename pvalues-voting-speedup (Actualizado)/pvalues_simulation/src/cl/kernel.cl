//#define LINTER
#ifdef LINTER
#include <opencl-c.h>
#define __kernel static
#define __constant
#define __global
#error "Undefine LINTER"
#endif

#include "pvalues-simulation/src/cl/common_cl.h"

#if defined(USE_DOUBLE_PRECISION)
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif
#endif

#include "pvalues-simulation/src/cl/pcg.cl"

typedef long log_t;
typedef struct __attribute__((__packed__)) {real_t prob; short alias;} alias_pair;

__kernel void sample_alias(
        __global alias_pair* pairs,
        const short n0, const short c, const log_t beta_n,
        const ulong trial_min, const ulong trial_max, const ulong step,
        __global ulong* result,
        __constant log_t* log_p, __constant log_t* lgac
) {

    const ulong index = get_global_id(0);
    const ulong start = trial_min + index*step;
    const ulong end = start + step > trial_max ? trial_max : start + step;

    const short N = n0+1;
    const ulong N2 = N*N;

    short n;
    ulong base;
    log_t beta_s;
    pcg32_random_t rng;

    short res;
    real_t x;

    ulong total = 0;
    for (ulong trial=start; trial<end; trial++) {
        n = n0;
        base = 0;
        beta_s = 0;
        pcg32_srandom_r(&rng, trial, trial);

        for (short i=0; i<c-1; i++) {
            x = (n+1)*(pcg32_random_r(&rng) * FACTOR32);
            res = (short)x;

            alias_pair pair = pairs[base + N*n + res];
            res += (x >= pair.prob)*(pair.alias - res);

            n -= res;
            base += N2;
            beta_s += res*log_p[i] - lgac[res];
        }

        beta_s += n*log_p[c-1] - lgac[n];
        total += (beta_s <= beta_n);
    }
    result[index] = total;
}
