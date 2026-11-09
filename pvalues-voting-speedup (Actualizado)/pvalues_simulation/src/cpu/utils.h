#pragma once

#ifdef _MSC_VER
#include <Windows.h>
#else
#include <time.h>
//#define CLOCK CLOCK_REALTIME
#define CLOCK CLOCK_MONOTONIC
#endif

static double ipow(double base, size exp) {
    double result = 1.0;
    while (1) {
        if (exp & 1) result *= base;
        exp >>= 1;
        if (!exp) break;
        base *= base;
    }
    return result;
}

static void print_instance(small_t n, small_t c, const size* real, const double* probs) {
    printf("n: %d\n", n);
    printf("c: %d\n", c);
    printf("r: ["); for (small_t i=0; i < c; i++) printf("%lld, ", real[i]); printf("]\n");
    printf("p: ["); for (small_t i=0; i < c; i++) printf("%f, ", probs[i]); printf("]\n");
}

static void print_results(output* out, double elapsed) {
    printf("total time: %f\n", elapsed);
    printf("trials = %lld\n", out->trials);
    printf("chi test = %.9f\n", out->p_value_chisq);
    printf("LLR test = %.9f\n", out->p_value_llr);
    printf("p-value  = %.9f\n", out->p_value);
}

double seconds() {
#ifdef _MSC_VER
    static LARGE_INTEGER frequency;
    if (frequency.QuadPart == 0) QueryPerformanceFrequency(&frequency);

    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec now;
    clock_gettime(CLOCK, &now);
    return now.tv_sec + now.tv_nsec/1e9;
#endif
}
