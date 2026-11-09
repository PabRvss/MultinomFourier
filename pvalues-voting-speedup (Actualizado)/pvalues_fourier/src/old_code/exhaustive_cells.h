#pragma once

#include <stdatomic.h>

static void iterative_cells_core(
        const int_t N,
        const int_t K,
        const cell_t* matrix,
        real_t* restrict p_acc,
        const int_t k_start,
        const int_t N_remain_start,
        const real_t r_acc_start,
        const real_t logp_acc_start
) {
    if (k_start == K-1) {  // only the last category remains: it takes whatever is left
        const int_t idx = (K-1)*(N+1) + N_remain_start;
        const real_t r_final = r_acc_start + matrix[idx].reward;
        const real_t logp_final = logp_acc_start + matrix[idx].log_prob;
        if (r_final <= 0) *p_acc = *p_acc + EXP(MAX_EXP + logp_final);
        return;
    }

    const int_t depth = K - k_start;  // stack levels k_start..K-1; levels k_start..K-2 loop
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
        int_t index_k = k*(N+1);
        const int_t n = n_cur[d];
        const real_t r_next = r_acc[d] + matrix[index_k + n].reward;
        const real_t logp_next = logp_acc[d] + matrix[index_k + n].log_prob;
        const int_t N_next = N_remain[d] - n;

        if (k == K-2) {  // last category is always the remainder
            const int_t idx_last = (K-1)*(N+1) + N_next;
            const real_t r_final = r_next + matrix[idx_last].reward;
            const real_t logp_final = logp_next + matrix[idx_last].log_prob;
            if (r_final <= 0) *p_acc = *p_acc + EXP(MAX_EXP + logp_final);
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
    int_t N, K;
    const cell_t* restrict matrix;
    real_t logp_acc;
    atomic_int_fast64_t* restrict next_n0;  // shared work queue: next category-0 count to claim
    real_t p_acc;  // thread-local partial sum, written once before the thread exits
} iterative_cells_args_t;

static void* iterative_cells_thread(void* args_void) {
    iterative_cells_args_t* args = args_void;
    const int_t N = args->N;
    real_t p_acc = 0;
    int_t n0;
    while ((n0 = atomic_fetch_add_explicit(args->next_n0, 1, memory_order_relaxed)) <= N) {
        const real_t r0 = args->matrix[n0].reward;
        const real_t logp0 = args->logp_acc + args->matrix[n0].log_prob;
        iterative_cells_core(args->N, args->K, args->matrix, &p_acc, 1, N-n0, r0, logp0);
    }

    args->p_acc = p_acc;
    return NULL;
}

static void iterative_cells_parallel(
        const int_t N,
        const int_t K,
        const cell_t* restrict matrix,
        real_t logp_acc,
        real_t* restrict p_acc,
        const int_t threads
) {
    const int_t n_threads = threads > N+1 ? N+1 : threads;  // no point over-subscribing n0 in [0, N]

    atomic_int_fast64_t next_n0;
    atomic_init(&next_n0, 0);

    pthread_t* tid = malloc(n_threads*sizeof(pthread_t));
    iterative_cells_args_t* args = malloc(n_threads*sizeof(iterative_cells_args_t));

    for (int_t t=0; t<n_threads; t++) {
        args[t].N = N;
        args[t].K = K;
        args[t].matrix = matrix;
        args[t].logp_acc = logp_acc;
        args[t].next_n0 = &next_n0;
        args[t].p_acc = 0;
        pthread_create(&tid[t], NULL, iterative_cells_thread, &args[t]);
    }

    for (int_t t=0; t<n_threads; t++) pthread_join(tid[t], NULL);
    for (int_t t=0; t<n_threads; t++) *p_acc += args[t].p_acc;

    free(tid);
    free(args);
}

static void exhaustive_cells(multinomial* mult, multinomial_result* mult_res, options_t* options) {
    const int_t N = mult->N;
    const int_t K = mult->K;
    real_t p_acc = 0;
    const real_t logp_acc = lgac[N];
    if (options->threads > 1) iterative_cells_parallel(N, K, mult->matrix, logp_acc, &p_acc, options->threads);
    else iterative_cells_core(N, K, mult->matrix, &p_acc, 0, N, 0, logp_acc);
    p_acc = p_acc*EXP(-MAX_EXP);
    if (options->verbose) printf("exhaustive = %.8e\n", p_acc);
    mult_res->pval = p_acc;
    mult_res->converged = 1;
    mult_res->terms = -1;
}
