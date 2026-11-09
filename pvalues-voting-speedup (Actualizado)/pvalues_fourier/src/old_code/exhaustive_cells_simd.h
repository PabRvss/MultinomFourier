#pragma once
#include <stdatomic.h>

// SIMD variant of iterative_cells_core (exhaustive_cells.h): same explicit-stack composition
// enumeration reading precomputed cell_t reward/log_prob values -- fill_rewards/fill_probs already
// did every reward_power/pow/log/exp call once, up front, so the leaf loop here is pure loads and
// adds. That's the key difference from exhaustive_simd.h (the SIMD variant of iterative_core, which
// still calls reward_power/reward_logp fresh per node): there, each SIMD lane had to be filled by a
// per-lane scalar loop no matter what, since the source values come from function calls. Here the
// source values are already sitting in memory, so the "k == K-2" batched level below loads STRIDE
// consecutive reward/log_prob values for the looping category k with a single, genuine contiguous
// vector load -- but only once they're repacked into local SoA (struct-of-arrays) buffers first.
// mult->matrix stays cell_t (AoS: log_prob, reward, shift interleaved), since the Fourier/
// saddlepoint path (fill_interval, eval_gamma_thread, fill_cells) still relies on that layout and
// STRIDE consecutive cells along an AoS row would be a strided gather, not a contiguous load. The
// SoA repack is local to exhaustive_cells_simd()'s call and never touches mult->matrix itself.
//
// The complementary last-category (K-1) lookup runs in the opposite direction as n increases
// (N_next = N_remain[d]-n decreases), so although it's still a contiguous run of memory, it's
// descending -- loading that as a vector would need a reverse permute, not worth it for a load with
// no transcendental math behind it either way, so it stays a per-lane scalar loop, same tradeoff
// exhaustive_simd.h already makes for the pieces that can't be expressed as plain vec_t arithmetic.
//
// Comparison stays the exact `<= 0` exhaustive_cells.h uses (not `<= r0` like exhaustive_simd.h):
// mult->matrix's reward field is already shifted by -rk in fill_rewards, so the summed leaf reward
// is (actual reward - r0) and the observed composition's own leaf cancels to exactly 0 by
// construction (see fill_rewards's `i == x0[k] ? rk : ...`), not by relying on cancellation here.
#ifdef USE_SIMD
#ifdef USE_DOUBLE_PRECISION
#define VEC_LOADU(ptr) _mm512_loadu_pd(ptr)
#else
#define VEC_LOADU(ptr) _mm512_loadu_ps(ptr)
#endif
#endif

static void iterative_cells_core_simd(
        const int_t N,
        const int_t K,
        const real_t* restrict reward,
        const real_t* restrict log_prob,
        real_t* restrict p_acc,
        const int_t k_start,
        const int_t N_remain_start,
        const real_t r_acc_start,
        const real_t logp_acc_start
) {
    if (k_start == K-1) {  // only the last category remains: it takes whatever is left
        const int_t idx = (K-1)*(N+1) + N_remain_start;
        const real_t r_final = r_acc_start + reward[idx];
        const real_t logp_final = logp_acc_start + log_prob[idx];
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

        if (k == K-2) {  // last category is always the remainder: batch this level's whole range
            const int_t index_k = k*(N+1);
            const int_t index_last = (K-1)*(N+1);
            const real_t r_acc_d = r_acc[d];
            const real_t logp_acc_d = logp_acc[d];
            const int_t N_remain_d = N_remain[d];
            real_t p_leaf = 0;

#ifdef USE_SIMD
            const vec_t r_acc_vec = r_acc_d*VEC_ONE;
            const vec_t logp_acc_vec = logp_acc_d*VEC_ONE;

            int_t n = n_cur[d];
            for (; n + STRIDE - 1 <= N_remain_d; n += STRIDE) {
                const vec_t reward_vec = VEC_LOADU(&reward[index_k + n]);
                const vec_t logp_vec = VEC_LOADU(&log_prob[index_k + n]);

                vec_t reward_last_vec, logp_last_vec;
                for (int s=0; s<STRIDE; s++) {
                    const int_t N_next = N_remain_d - (n + s);
                    reward_last_vec[s] = reward[index_last + N_next];
                    logp_last_vec[s] = log_prob[index_last + N_next];
                }

                const vec_t r_final_vec = r_acc_vec + reward_vec + reward_last_vec;
                const vec_t logp_final_vec = logp_acc_vec + logp_vec + logp_last_vec;

                // Final compare + EXP accumulation stays a per-lane scalar loop, same as
                // iterative_core_simd (exhaustive_simd.h): no masked-add primitive in this
                // codebase's SIMD macros, so there's nothing to gain vectorizing this part.
                for (int s=0; s<STRIDE; s++) {
                    if (r_final_vec[s] <= 0) p_leaf += EXP(MAX_EXP + logp_final_vec[s]);
                }
            }
            for (; n <= N_remain_d; n++) {  // scalar tail: partial STRIDE batch at the end of the range
                const int_t N_next = N_remain_d - n;
                const real_t r_final = r_acc_d + reward[index_k + n] + reward[index_last + N_next];
                const real_t logp_final = logp_acc_d + log_prob[index_k + n] + log_prob[index_last + N_next];
                if (r_final <= 0) p_leaf += EXP(MAX_EXP + logp_final);
            }
#else
            for (int_t n = n_cur[d]; n <= N_remain_d; n++) {
                const int_t N_next = N_remain_d - n;
                const real_t r_final = r_acc_d + reward[index_k + n] + reward[index_last + N_next];
                const real_t logp_final = logp_acc_d + log_prob[index_k + n] + log_prob[index_last + N_next];
                if (r_final <= 0) p_leaf += EXP(MAX_EXP + logp_final);
            }
#endif
            // Folded as one local sum then added once, instead of accumulating each term into
            // *p_acc individually: mathematically identical to iterative_cells_core, but not
            // guaranteed bit-identical since floating-point addition isn't associative (same
            // harmless class of reordering exhaustive_simd.h's iterative_core_simd already uses).
            *p_acc = *p_acc + p_leaf;

            d--;
            if (d >= 0) n_cur[d]++;
            continue;
        }

        const int_t n = n_cur[d];
        const int_t index_k = k*(N+1);
        const real_t r_next = r_acc[d] + reward[index_k + n];
        const real_t logp_next = logp_acc[d] + log_prob[index_k + n];
        const int_t N_next = N_remain[d] - n;

        d++;  // descend
        N_remain[d] = N_next;
        r_acc[d] = r_next;
        logp_acc[d] = logp_next;
        n_cur[d] = 0;
    }

    free(N_remain);
    free(r_acc);
    free(logp_acc);
    free(n_cur);
}

typedef struct {
    int_t N, K;
    const real_t* restrict reward;
    const real_t* restrict log_prob;
    real_t logp_acc;
    atomic_int_fast64_t* restrict next_n0;  // shared work queue: next category-0 count to claim
    real_t p_acc;  // thread-local partial sum, written once before the thread exits
} iterative_cells_simd_args_t;

static void* iterative_cells_thread_simd(void* args_void) {
    iterative_cells_simd_args_t* args = args_void;
    const int_t N = args->N;
    real_t p_acc = 0;
    int_t n0;
    while ((n0 = atomic_fetch_add_explicit(args->next_n0, 1, memory_order_relaxed)) <= N) {
        const real_t r0 = args->reward[n0];
        const real_t logp0 = args->logp_acc + args->log_prob[n0];
        iterative_cells_core_simd(args->N, args->K, args->reward, args->log_prob, &p_acc, 1, N-n0, r0, logp0);
    }

    args->p_acc = p_acc;
    return NULL;
}

static void iterative_cells_parallel_simd(
        const int_t N,
        const int_t K,
        const real_t* restrict reward,
        const real_t* restrict log_prob,
        real_t logp_acc,
        real_t* restrict p_acc,
        const int_t threads
) {
    if (K == 1 || threads <= 1 || N == 0) {  // no category-0 split to parallelize over
        iterative_cells_core_simd(N, K, reward, log_prob, p_acc, 0, N, 0, logp_acc);
        return;
    }

    const int_t n_threads = threads > N+1 ? N+1 : threads;  // no point over-subscribing n0 in [0, N]

    atomic_int_fast64_t next_n0;
    atomic_init(&next_n0, 0);

    pthread_t* tid = malloc(n_threads*sizeof(pthread_t));
    iterative_cells_simd_args_t* args = malloc(n_threads*sizeof(iterative_cells_simd_args_t));

    for (int_t t=0; t<n_threads; t++) {
        args[t].N = N;
        args[t].K = K;
        args[t].reward = reward;
        args[t].log_prob = log_prob;
        args[t].logp_acc = logp_acc;
        args[t].next_n0 = &next_n0;
        args[t].p_acc = 0;
        pthread_create(&tid[t], NULL, iterative_cells_thread_simd, &args[t]);
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

static void exhaustive_cells_simd(multinomial* mult, multinomial_result* mult_res, options_t* options) {
    const int_t N = mult->N;
    const int_t K = mult->K;
    const int_t matrix_size = K*(N+1);

    real_t* restrict reward = malloc(matrix_size*sizeof(real_t));
    real_t* restrict log_prob = malloc(matrix_size*sizeof(real_t));
    for (int_t idx=0; idx<matrix_size; idx++) {
        reward[idx] = mult->matrix[idx].reward;
        log_prob[idx] = mult->matrix[idx].log_prob;
    }

    real_t p_acc = 0;
    const real_t logp_acc = lgac[N];
    if (options->threads > 1) iterative_cells_parallel_simd(N, K, reward, log_prob, logp_acc, &p_acc, options->threads);
    else iterative_cells_core_simd(N, K, reward, log_prob, &p_acc, 0, N, 0, logp_acc);
    p_acc = p_acc*EXP(-MAX_EXP);
    if (options->verbose) printf("exhaustive cells SIMD = %.8e\n", p_acc);
    mult_res->pval = p_acc;
    mult_res->converged = 1;
    mult_res->terms = -1;

    free(reward);
    free(log_prob);
}