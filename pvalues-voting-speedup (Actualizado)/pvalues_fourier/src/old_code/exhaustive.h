#pragma once
#include <stdatomic.h>

static void iterative_core(
        const real_t lambda,
        const int_t N,
        const int_t K,
        const real_t* restrict log_probs,
        real_t* restrict p_acc,
        const real_t r0,
        const int_t k_start,
        const int_t N_remain_start,
        const real_t r_acc_start,
        const real_t logp_acc_start
) {
    if (k_start == K-1) {  // only the last category remains: it takes whatever is left
        const real_t log_p = log_probs[K-1];
        const real_t r_final = r_acc_start + reward_power(N, K-1, log_p, N_remain_start, lambda);
        const real_t logp_final = logp_acc_start + reward_logp(log_p, N_remain_start);
        if (r_final <= r0) *p_acc = *p_acc + EXP(MAX_EXP + logp_final);
        return;
    }

    const int_t depth = K - k_start;
    int_t* restrict N_remain = malloc(depth*sizeof(int_t));
    real_t* restrict r_acc = malloc(depth*sizeof(real_t));
    real_t* restrict logp_acc = malloc(depth*sizeof(real_t));
    int_t* restrict n_cur = malloc((depth-1)*sizeof(int_t));

    int_t d = 0;  // depth relative to k_start; absolute category is k_start+d
    N_remain[0] = N_remain_start;
    r_acc[0] = r_acc_start;
    logp_acc[0] = logp_acc_start;
    n_cur[0] = 0;

    while (d >= 0) {
        if (n_cur[d] > N_remain[d]) {  // exhausted this level, backtrack
            d--;
            if (d >= 0) n_cur[d]++;
            continue;
        }

        const int_t k = k_start + d;
        const int_t n = n_cur[d];
        const real_t log_p = log_probs[k];
        const real_t r_next = r_acc[d] + reward_power(N, k, log_p, n, lambda);
        const real_t logp_next = logp_acc[d] + reward_logp(log_p, n);
        const int_t N_next = N_remain[d] - n;

        if (k == K-2) {  // last category is always the remainder
            const real_t log_p_last = log_probs[K-1];
            const real_t r_final = r_next + reward_power(N, K-1, log_p_last, N_next, lambda);
            const real_t logp_final = logp_next + reward_logp(log_p_last, N_next);
            if (r_final <= r0) *p_acc = *p_acc + EXP(MAX_EXP + logp_final);
            n_cur[d]++;
        } else {  // descend
            d++;
            N_remain[d] = N_next;
            r_acc[d] = r_next;
            logp_acc[d] = logp_next;
            n_cur[d] = 0;
        }
    }

    free(N_remain);
    free(r_acc);
    free(logp_acc);
    free(n_cur);
}

typedef struct {
    real_t lambda;
    int_t N, K;
    const real_t* restrict log_probs;
    real_t r0;
    real_t logp_acc0;
    atomic_int_fast64_t* restrict next_n0;  // shared work queue: next category-0 count to claim
    real_t p_acc;  // thread-local partial sum, written once before the thread exits
} iterative_args_t;

static void* iterative_thread(void* args_void) {
    iterative_args_t* args = (iterative_args_t*)args_void;
    const int_t N = args->N;
    const real_t lambda = args->lambda;
    const real_t* restrict log_probs = args->log_probs;

    real_t p_acc = 0;
    int_t n0;
    while ((n0 = atomic_fetch_add_explicit(args->next_n0, 1, memory_order_relaxed)) <= N) {
        const real_t log_p0 = log_probs[0];
        const real_t r1 = reward_power(N, 0, log_p0, n0, lambda);
        const real_t logp1 = args->logp_acc0 + reward_logp(log_p0, n0);
        iterative_core(lambda, N, args->K, log_probs, &p_acc, args->r0, 1, N-n0, r1, logp1);
    }

    args->p_acc = p_acc;
    return NULL;
}

static void iterative_parallel(
        const real_t lambda,
        const int_t N,
        const int_t K,
        const real_t* restrict log_probs,
        real_t* restrict p_acc,
        const real_t r0,
        const real_t logp_acc0,
        const int_t threads
) {
    if (K == 1 || threads <= 1 || N == 0) {  // no category-0 split to parallelize over
        iterative_core(lambda, N, K, log_probs, p_acc, r0, 0, N, 0, logp_acc0);
        return;
    }

    const int_t n_threads = threads > N+1 ? N+1 : threads;  // no point over-subscribing n0 in [0, N]

    atomic_int_fast64_t next_n0;
    atomic_init(&next_n0, 0);

    pthread_t* tid = malloc(n_threads*sizeof(pthread_t));
    iterative_args_t* args = malloc(n_threads*sizeof(iterative_args_t));

    for (int_t t=0; t<n_threads; t++) {
        args[t].lambda = lambda;
        args[t].N = N;
        args[t].K = K;
        args[t].log_probs = log_probs;
        args[t].r0 = r0;
        args[t].logp_acc0 = logp_acc0;
        args[t].next_n0 = &next_n0;
        args[t].p_acc = 0;
        pthread_create(&tid[t], NULL, iterative_thread, &args[t]);
    }

    real_t total = 0;
    for (int_t t=0; t<n_threads; t++) {
        pthread_join(tid[t], NULL);
        total += args[t].p_acc;
    }
    *p_acc = total;

    free(tid);
    free(args);
}

static void exhaustive(multinomial* mult, multinomial_result* mult_res, options_t* options) {
    const real_t lambda = options->lambda;
    const int_t N = mult->N;
    const int_t K = mult->K;
    const real_t r0 = mult->r0;
    real_t p_acc = 0;
    const real_t logp_acc = lgac[N];
    if (options->threads > 1) iterative_parallel(lambda, N, K, mult->log_probs, &p_acc, r0, logp_acc, options->threads);
    else iterative_core(lambda, N, K, mult->log_probs, &p_acc, r0, 0, N, 0, logp_acc);
    p_acc = p_acc*EXP(-MAX_EXP);
    if (options->verbose) printf("exhaustive = %.8e\n", p_acc);
    mult_res->pval = p_acc;
    mult_res->converged = 1;
    mult_res->terms = -1;
}
