#pragma once

#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define HAVE_STRUCT_TIMESPEC
#include <pthread.h>

#ifdef _WIN32
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef short int16_t;
#endif

typedef int64_t pyint;
typedef int64_t size;
typedef int16_t small_t;

// Compilation parameters
//#define PARALLEL_NUMPY
#define PARALLEL_ALIAS

#define CATS_FREQ 5

// External parameters
static bool verbose = false;
static small_t threads = 20;

static small_t n_max = (small_t) 1e3;
static const small_t c_max = (small_t) 1e2;

static double threshold = 1e-4;
static double error= 1e-10;
static double confidence = 0.95;
static double relative_error = 1e1;

static bool quick = true;
static bool estimate = true;

static size trials_numpy = 1000;
static size *trials_array;

// Dependent parameters
static size trials_n;
static double* p_crit_array;
double z_confidence;
static double *lgac;

typedef struct {
    double p_value;
    double p_value_chisq;
    double p_value_llr;
    size trials;
} output;

#if defined(__GNUC__)
#define EXTERNAL __attribute__((visibility("default")))
#define PACK( __Declaration__ ) __Declaration__ __attribute__((__packed__))
#elif defined(_MSC_VER)
#define EXTERNAL __declspec(dllexport)
#define PACK( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop))
#endif

#ifdef USE_OPENCL
#include "cl/common_cl.h"
#include "cl/base_cl.h"
#endif

static void calculate_lgac() {
    size lgac_n = n_max+c_max+1;
    lgac = (double*)malloc(lgac_n*sizeof(double));
    lgac[0] = 0.0;

    double s = 0.0;
    double d = 0.0;
    for (small_t i=1; i <= n_max + c_max; i++) {
        double l = log(i);

        double y = l - d;
        double t = s + y;
        d = (t - s) - y;
        s = t;
        lgac[i] = s;
    }

#ifdef USE_OPENCL
    lgac_cl = (log_t*)malloc(lgac_n*sizeof(log_t));
    for (size i=0; i<lgac_n; i++) lgac_cl[i] = (log_t) (BIGM * lgac[i]);
#endif
}

static void prepare_cl() {
#ifdef USE_OPENCL
    load_kernel();
#endif
}

static double inverse_normal(double p);
static double ipow(double base, size exp);
static void calculate_pcrits() {
    z_confidence = inverse_normal((1+confidence)/2);

    p_crit_array = (double*)malloc(trials_n*sizeof(double));

    double p = threshold;
    double q = 1.0-p;
    double w = inverse_normal(1-error);
    for (size s=0; s<trials_n; s++) {
        size n = trials_array[s];
        size N = n+1;
        double nd = (double)n;
        double test = nd*p/q;
        double p_crit = 1.0;

        if (test >= 9) {
            double sigma = sqrt(nd*p*q);
            p_crit = (w*sigma + nd*p + 0.5)/nd;
        } else {
            double d = 1.0/q;
            double t = ipow(q, n);
            double cdf = t;
            for (size i=1; i<N; i++) {
                if (cdf > 1-error) {
                    p_crit = (double)(i-1)/(double)n;
                    break;
                }
                t *= p*d*(double)(N-i)/(double)i;
                cdf += t;
            }
        }
        p_crit_array[s] = p_crit;
    }
}

static void print_params() {
    printf("parameters:\n\n");

    printf("verbose: %d\n", verbose);
    printf("threads: %d\n", threads);
    printf("n_max: %d\n", n_max);
    printf("c_max: %d\n", c_max);
    printf("\n");
    printf("log10 threshold: %.3f\n", log10(threshold));
    printf("log10 error: %.3f\n", log10(error));
    printf("p-value confidence: %f\n", confidence);
    printf("p-value relative error: %f\n", relative_error);
    printf("\n");
    printf("quick: %d\n", quick);
    printf("trials_numpy: %lld\n", trials_numpy);
    printf("trials_array: ["); for (size i=0; i<trials_n; i++) printf("%lld, ", trials_array[i]); printf("]\n");
    printf("\n");
}

static void init_default() {
    int i1=2; int i2=7;
    trials_n = (i2-i1+1);
    trials_array = (size*)malloc(trials_n*sizeof(size));
    for (size i=0; i<trials_n; i++) trials_array[i] = (size) round(pow(10, (double)i+i1));

    calculate_lgac();
    calculate_pcrits();
    print_params();
    prepare_cl();
}

EXTERNAL void init_ext(
        pyint threads1, pyint n_max1,
        double threshold1, double error1,
        double confidence1, double relative_error1,
        pyint quick1, pyint trials_n1, pyint trials_numpy1,
        const size *trials_array1, pyint verbose1
) {
    verbose = (bool)verbose1;
    threads = (small_t)threads1;
    n_max = (small_t)n_max1;

    threshold = (double)threshold1;
    error = (double)error1;
    confidence = (double)confidence1;
    relative_error = (double)relative_error1;

    quick = (bool)quick1;
    trials_n = (size)trials_n1;
    trials_numpy = (size)trials_numpy1;

    trials_array = (size*)malloc(trials_n*sizeof(size));
    for (size i=0; i<trials_n; i++) trials_array[i] = trials_array1[i];

    calculate_lgac();
    calculate_pcrits();
    print_params();
    prepare_cl();
}

EXTERNAL void finish() {
#ifdef USE_OPENCL
    free_kernel();
    free(lgac_cl);
#endif
    free(lgac);
    free(trials_array);
    free(p_crit_array);
}
