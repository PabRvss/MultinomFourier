#pragma once

static inline size sample_multi_numpy(
        small_t n, small_t c,
        const double* probs, double beta_n, const double* lowp_remain,
        size trial, const double* log_p, pcg32_random_t* rng
        ) {

    double beta_s = 0.0;
    double p_remain = 1.0;
    small_t res;
    pcg32_srandom_r(rng, trial, trial);
    for (small_t i=0; i<c-1; i++) {
        double p = probs[i];
        res = random_binomial(p/p_remain, n, rng);
        n -= res;
        p_remain -= p;
        beta_s += res*log_p[i] - lgac[res];

        if (beta_s <= beta_n) return 1;
        double bound_lo = n*lowp_remain[i+1] - lgac[n];
        if (beta_s + bound_lo > beta_n) return 0;
    }
    beta_s += n*log_p[c-1] - lgac[n];
    return beta_s <= beta_n;
}

typedef struct {
    const double* probs;
    const double* log_p;
    const double* lowp_remain;
    double beta_n;

    size start;
    size end;

    size res;

    small_t n;
    small_t c;
} arguments_numpy;

static void* sample_multi_numpy_thread(void* args_void) {
    arguments_numpy* args = (arguments_numpy*)args_void;

    small_t n = args->n;
    small_t c = args->c;

    double beta_n = args->beta_n;

    size start = args->start;
    size end = args->end;

    size res = 0;
    pcg32_random_t rng;
    for (size i=start; i<end; i++) res += sample_multi_numpy(n, c, args->probs, beta_n, args->lowp_remain, i, args->log_p, &rng);
    args->res = res;
    return 0;
}

static size sample_multi_numpy_parallel(
        small_t n, small_t c,
        const double* probs, const double* log_p, const double* lowp_remain,
        double beta_n, size trial_min, size trial_max
        ) {

    pthread_t* tid = (pthread_t*)malloc(threads*sizeof(pthread_t));
    arguments_numpy* args_array = (arguments_numpy*)malloc(threads * sizeof(arguments_numpy));

    size step = (trial_max-trial_min+threads-1)/threads;
    for (small_t i=0; i<threads; i++) {
        size start = trial_min + i*step;
        size end = start + step;
        if (end > trial_max) end = trial_max;

        arguments_numpy* args_i = &args_array[i];

        args_i->n = n;
        args_i->c = c;

        args_i->probs = probs;
        args_i->log_p = log_p;
        args_i->lowp_remain = lowp_remain;
        args_i->beta_n = beta_n;

        args_i->start = start;
        args_i->end = end;

        pthread_create(&tid[i], NULL, sample_multi_numpy_thread, args_i);
    }
    for (small_t i=0; i<threads; i++) pthread_join(tid[i], NULL);

    size res = 0;
    for (small_t i=0; i<threads; i++) res += args_array[i].res;

    free(tid);
    free(args_array);

    return res;
}
