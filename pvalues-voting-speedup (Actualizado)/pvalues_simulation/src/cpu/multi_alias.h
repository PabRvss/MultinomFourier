#pragma once

#include "alias.h"

typedef struct {
    alias_pair* pairs;

    small_t n;
    small_t c;
    size mem;
} multi_alias;

static void free_multi_alias(const multi_alias* mult) {
    free(mult->pairs);
}

typedef struct {
    const multi_alias* mult;
    const double* log_p;
    const double* lowp_remain;
    double beta_n;
    size start;
    size end;
    size res;
} arguments_alias;

static void create_multi_alias(multi_alias* mult, const small_t n, const small_t c, const double* probs) {
    const small_t N = n + 1;
    alias_pair* pairs = (alias_pair*)malloc((c-1)*N*N*sizeof(alias_pair));

    mult->n = n;
    mult->c = c;
    mult->pairs = pairs;

    double p_remain = 1.0;
    for (small_t i=0; i < c - 1; i++) {
        double pi = probs[i];
        double pir = pi/p_remain;
        for (small_t j=0; j < N; j++) create_binomial_table(&pairs[N * N * i + N * j], j, pir);
        p_remain -= pi;
    }
}

static inline size sample_multi_alias(
        const small_t N, small_t n, const small_t c,
        const size trial, pcg32_random_t* rng,
        const alias_pair* pairs, const double* log_p, const double* lowp_remain, const double beta_n
        ) {
    small_t res;
    double beta_s = 0.0;
    double x;

    pcg32_srandom_r(rng, trial, trial);
    uint32_t N2 = N*N;
    uint32_t base = N*n;

    for (small_t i=0; i < c - 1; i++) {
        x = (n+1)*pcg32_random_double(rng);
        res = (small_t)x;

        const alias_pair pair = pairs[base + res];
        res += (x >= pair.prob)*(pair.alias - res);

        n -= res;
        base += N2 - N*res;
        beta_s += res*log_p[i] - lgac[res];

        if (i % CATS_FREQ == CATS_FREQ-1) {
            if (beta_s <= beta_n) return 1;
            double bound_lo = n*lowp_remain[i+1] - lgac[n];
            if (beta_s + bound_lo > beta_n) return 0;
        }
    }

    beta_s += n*log_p[c-1] - lgac[n];
    return beta_s <= beta_n;
}

static void* sample_multi_alias_thread(void* args_void) {
    arguments_alias* args = args_void;

    const multi_alias* mult = args->mult;
    const small_t n = mult->n;
    const small_t c = mult->c;
    const alias_pair* pairs = mult->pairs;
    const double* log_p = args->log_p;
    const double* lowp_remain = args->lowp_remain;
    const double beta_n = args->beta_n;

    const small_t N = n + 1;

    size res = 0;
    pcg32_random_t rng;
    for (size i=args->start; i<args->end; i++) res += sample_multi_alias(N, n, c, i, &rng, pairs, log_p, lowp_remain, beta_n);
    args->res = res;
    return 0;
}

static size sample_multi_alias_parallel(
        const multi_alias* mult,
        const double beta_n, const double* log_p, const double* lowp_remain,
        const size trial_min, const size trial_max
        ) {

    pthread_t* tid = (pthread_t*)malloc(threads*sizeof(pthread_t));
    arguments_alias* args_array = (arguments_alias*)malloc(threads * sizeof(arguments_alias));

    size step = (trial_max-trial_min+threads-1)/threads;
    for (small_t i=0; i < threads; i++) {
        size start = trial_min + i*step;
        size end = start + step;
        if (end > trial_max) end = trial_max;

        arguments_alias* args_i = &args_array[i];
        args_i->mult = mult;
        args_i->start = start;
        args_i->end = end;
        args_i->beta_n = beta_n;
        args_i->log_p = log_p;
        args_i->lowp_remain = lowp_remain;

        pthread_create(&tid[i], NULL, sample_multi_alias_thread, args_i);
    }
    for (small_t i=0; i < threads; i++) pthread_join(tid[i], NULL);

    size res = 0;
    for (size i=0; i<threads; i++) res += args_array[i].res;

    free(tid);
    free(args_array);

    return res;
}
