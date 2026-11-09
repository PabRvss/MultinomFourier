#pragma once
#include <stdatomic.h>

// Further-optimized variant of exhaustive_cells_simd.h's iterative_cells_core_simd, targeting
// overhead that survives once the leaf loop is already SIMD-batched over precomputed SoA
// reward[]/log_prob[] arrays. Four ideas, each behind a #define so it can be tuned per
// hardware/compiler without touching code:
//
// - CELLS_OPT_MAX_DEPTH: iterative_cells_core_simd (exhaustive_cells_simd.h) mallocs its four
//   O(K)-sized scratch arrays (N_remain/r_acc/logp_acc/n_cur) FRESH for every n0 sub-problem --
//   N+1 malloc/free round-trips per exhaustive_cells_simd_opt() call. K is always small in this
//   codebase's actual usage (a handful of categories), so each worker thread instead carries one
//   fixed-size stack buffer, sized to this bound, reused across every n0 it claims; only if K
//   exceeds the bound does it fall back to a per-thread (not per-n0) heap allocation.
// - CELLS_OPT_CHUNK: the shared atomic work queue hands out CELLS_OPT_CHUNK consecutive n0
//   values per claim instead of one. For small K (e.g. K=3, where each n0's whole remaining
//   sub-problem is a single STRIDE-batchable range) a claim of 1 is barely any work relative to
//   the atomic op and thread dispatch around it; claiming a small contiguous run amortizes that
//   and improves locality, since adjacent n0 rows are adjacent in the SoA arrays.
// - CELLS_OPT_UNROLL: the k==K-2 leaf loop processes CELLS_OPT_UNROLL consecutive STRIDE-wide
//   vector loads per outer-loop pass instead of one, giving the compiler more independent
//   work per iteration (more ILP, fewer loop-control instructions per unit of actual work).
// - CELLS_OPT_PREFETCH_DIST: issues __builtin_prefetch CELLS_OPT_PREFETCH_DIST unrolled blocks
//   ahead of the block currently being processed (0 disables it). Included as a tunable rather
//   than assumed to help: reward[]/log_prob[] are K*(N+1) real_t each, which stays well within
//   L1/L2 for the problem sizes this method is actually used for (exhaustive enumeration is only
//   selected below options->enum_cutoff), so the expected win here is small to none in practice --
//   worth verifying empirically per hardware rather than leaving it un-tunable.
// - CELLS_OPT_VEC_EXP: the final compare+accumulate (EXP(MAX_EXP+logp_final) for lanes passing
//   r_final<=0) is a scalar per-lane loop by default, since it's the one piece with no masked-add
//   primitive in this codebase's SIMD macros -- exhaustive_simd.h found vectorizing an analogous
//   scalar step (reward_llr, which has a cheap xk==0 early-out) was a net loss once masking
//   overhead is counted. This EXP call has no such early-out (every passing lane needs it), so the
//   tradeoff is different: CELLS_OPT_VEC_EXP=1 batches it via sse_mathfun_generic's exp_vec across
//   the whole STRIDE-wide vector unconditionally, masking out non-passing lanes afterward instead
//   of skipping them -- a win only if enough lanes typically pass to outweigh always paying for
//   all of them. Worth measuring per instance/hardware rather than assuming either way.
//
// All five default to values chosen generically; scripts/benchmarks are expected to override them
// via -D when tuning for a specific machine/compiler.
#ifndef CELLS_OPT_MAX_DEPTH
#define CELLS_OPT_MAX_DEPTH 64
#endif

#ifndef CELLS_OPT_CHUNK
#define CELLS_OPT_CHUNK 4
#endif

#ifndef CELLS_OPT_UNROLL
#define CELLS_OPT_UNROLL 2
#endif

#ifndef CELLS_OPT_PREFETCH_DIST
#define CELLS_OPT_PREFETCH_DIST 0
#endif

#ifndef CELLS_OPT_VEC_EXP
#define CELLS_OPT_VEC_EXP 0
#endif

#ifdef USE_SIMD
#include "../sse_mathfun_generic/mathfun_generic.h"
#ifdef USE_DOUBLE_PRECISION
#define VEC_LOADU(ptr) _mm512_loadu_pd(ptr)
#else
#define VEC_LOADU(ptr) _mm512_loadu_ps(ptr)
#endif

// Sums EXP(MAX_EXP+logp_final) over lanes where r_final<=0. CELLS_OPT_VEC_EXP selects between the
// default scalar per-lane loop (skips EXP entirely for non-passing lanes) and a SIMD-batched
// exp_vec call (computes it for every lane, then masks -- see the CELLS_OPT_VEC_EXP comment above).
static inline real_t cells_leaf_batch_sum(const vec_t r_final_vec, const vec_t logp_final_vec) {
#if CELLS_OPT_VEC_EXP
    const vec_mask_t mask = VEC_CMP_MASK(r_final_vec, VEC_ZERO, SIMDE_CMP_LE_OQ);
    const vec_t exp_vals = exp_vec(MAX_EXP*VEC_ONE + logp_final_vec);
    const vec_t masked_vals = VEC_BLEND_REAL(mask, VEC_ZERO, exp_vals);
    real_t sum = 0;
    for (int s = 0; s < STRIDE; s++) sum += masked_vals[s];
    return sum;
#else
    real_t sum = 0;
    for (int s = 0; s < STRIDE; s++) {
        if (r_final_vec[s] <= 0) sum += EXP(MAX_EXP + logp_final_vec[s]);
    }
    return sum;
#endif
}
#endif

// Same traversal as iterative_cells_core_simd (exhaustive_cells_simd.h), except the four
// O(K)-sized scratch arrays are caller-provided (not malloc'd here) so a worker thread can
// allocate them once and reuse them across every n0 sub-problem it processes.
static void iterative_cells_core_simd_opt(
        const int_t N,
        const int_t K,
        const real_t* restrict reward,
        const real_t* restrict log_prob,
        real_t* restrict p_acc,
        const int_t k_start,
        const int_t N_remain_start,
        const real_t r_acc_start,
        const real_t logp_acc_start,
        int_t* restrict N_remain,   // scratch, size >= K
        real_t* restrict r_acc,     // scratch, size >= K
        real_t* restrict logp_acc,  // scratch, size >= K
        int_t* restrict n_cur       // scratch, size >= K
) {
    if (k_start == K-1) {  // only the last category remains: it takes whatever is left
        const int_t idx = (K-1)*(N+1) + N_remain_start;
        const real_t r_final = r_acc_start + reward[idx];
        const real_t logp_final = logp_acc_start + log_prob[idx];
        if (r_final <= 0) *p_acc = *p_acc + EXP(MAX_EXP + logp_final);
        return;
    }

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
            const int_t block = STRIDE*CELLS_OPT_UNROLL;

            int_t n = n_cur[d];
            for (; n + block - 1 <= N_remain_d; n += block) {
#if CELLS_OPT_PREFETCH_DIST > 0
                const int_t pf_n = n + CELLS_OPT_PREFETCH_DIST*block;
                if (pf_n + block - 1 <= N_remain_d) {
                    const int_t pf_N_next = N_remain_d - pf_n;
                    __builtin_prefetch(&reward[index_k + pf_n], 0, 1);
                    __builtin_prefetch(&log_prob[index_k + pf_n], 0, 1);
                    __builtin_prefetch(&reward[index_last + pf_N_next - block + 1], 0, 1);
                    __builtin_prefetch(&log_prob[index_last + pf_N_next - block + 1], 0, 1);
                }
#endif
                for (int u = 0; u < CELLS_OPT_UNROLL; u++) {
                    const int_t nb = n + u*STRIDE;
                    const vec_t reward_vec = VEC_LOADU(&reward[index_k + nb]);
                    const vec_t logp_vec = VEC_LOADU(&log_prob[index_k + nb]);

                    vec_t reward_last_vec, logp_last_vec;
                    for (int s = 0; s < STRIDE; s++) {
                        const int_t N_next = N_remain_d - (nb + s);
                        reward_last_vec[s] = reward[index_last + N_next];
                        logp_last_vec[s] = log_prob[index_last + N_next];
                    }

                    const vec_t r_final_vec = r_acc_vec + reward_vec + reward_last_vec;
                    const vec_t logp_final_vec = logp_acc_vec + logp_vec + logp_last_vec;

                    p_leaf += cells_leaf_batch_sum(r_final_vec, logp_final_vec);
                }
            }
            for (; n + STRIDE - 1 <= N_remain_d; n += STRIDE) {  // full-STRIDE tail (< 1 unrolled block)
                const vec_t reward_vec = VEC_LOADU(&reward[index_k + n]);
                const vec_t logp_vec = VEC_LOADU(&log_prob[index_k + n]);

                vec_t reward_last_vec, logp_last_vec;
                for (int s = 0; s < STRIDE; s++) {
                    const int_t N_next = N_remain_d - (n + s);
                    reward_last_vec[s] = reward[index_last + N_next];
                    logp_last_vec[s] = log_prob[index_last + N_next];
                }

                const vec_t r_final_vec = r_acc_vec + reward_vec + reward_last_vec;
                const vec_t logp_final_vec = logp_acc_vec + logp_vec + logp_last_vec;

                p_leaf += cells_leaf_batch_sum(r_final_vec, logp_final_vec);
            }
            for (; n <= N_remain_d; n++) {  // scalar tail (< 1 full STRIDE)
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
}

typedef struct {
    int_t N, K;
    const real_t* restrict reward;
    const real_t* restrict log_prob;
    real_t logp_acc;
    atomic_int_fast64_t* restrict next_n0;  // shared work queue: next chunk-start n0 to claim
    real_t p_acc;  // thread-local partial sum, written once before the thread exits
} iterative_cells_simd_opt_args_t;

static void* iterative_cells_thread_simd_opt(void* args_void) {
    iterative_cells_simd_opt_args_t* args = args_void;
    const int_t N = args->N;
    const int_t K = args->K;
    real_t p_acc = 0;

    int_t N_remain_stack[CELLS_OPT_MAX_DEPTH];
    real_t r_acc_stack[CELLS_OPT_MAX_DEPTH];
    real_t logp_acc_stack[CELLS_OPT_MAX_DEPTH];
    int_t n_cur_stack[CELLS_OPT_MAX_DEPTH];

    int_t* N_remain = N_remain_stack;
    real_t* r_acc = r_acc_stack;
    real_t* logp_acc = logp_acc_stack;
    int_t* n_cur = n_cur_stack;
    int_t* N_remain_heap = NULL;
    real_t* r_acc_heap = NULL;
    real_t* logp_acc_heap = NULL;
    int_t* n_cur_heap = NULL;

    if (K > CELLS_OPT_MAX_DEPTH) {  // rare: K beyond the stack bound, fall back to one heap alloc/thread
        N_remain_heap = malloc(K*sizeof(int_t));
        r_acc_heap = malloc(K*sizeof(real_t));
        logp_acc_heap = malloc(K*sizeof(real_t));
        n_cur_heap = malloc(K*sizeof(int_t));
        N_remain = N_remain_heap;
        r_acc = r_acc_heap;
        logp_acc = logp_acc_heap;
        n_cur = n_cur_heap;
    }

    int_t n0_start;
    while ((n0_start = atomic_fetch_add_explicit(args->next_n0, CELLS_OPT_CHUNK, memory_order_relaxed)) <= N) {
        const int_t n0_end = n0_start + CELLS_OPT_CHUNK - 1 > N ? N : n0_start + CELLS_OPT_CHUNK - 1;
        for (int_t n0 = n0_start; n0 <= n0_end; n0++) {
            const real_t r0 = args->reward[n0];
            const real_t logp0 = args->logp_acc + args->log_prob[n0];
            iterative_cells_core_simd_opt(args->N, K, args->reward, args->log_prob, &p_acc, 1, N-n0, r0, logp0,
                                           N_remain, r_acc, logp_acc, n_cur);
        }
    }

    free(N_remain_heap);
    free(r_acc_heap);
    free(logp_acc_heap);
    free(n_cur_heap);

    args->p_acc = p_acc;
    return NULL;
}

static void iterative_cells_parallel_simd_opt(
        const int_t N,
        const int_t K,
        const real_t* restrict reward,
        const real_t* restrict log_prob,
        real_t logp_acc,
        real_t* restrict p_acc,
        const int_t threads
) {
    if (K == 1 || threads <= 1 || N == 0) {  // no category-0 split to parallelize over
        int_t N_remain_stack[CELLS_OPT_MAX_DEPTH], n_cur_stack[CELLS_OPT_MAX_DEPTH];
        real_t r_acc_stack[CELLS_OPT_MAX_DEPTH], logp_acc_stack[CELLS_OPT_MAX_DEPTH];
        int_t* N_remain = N_remain_stack;
        real_t* r_acc = r_acc_stack;
        real_t* logp_acc_buf = logp_acc_stack;
        int_t* n_cur = n_cur_stack;
        if (K > CELLS_OPT_MAX_DEPTH) {
            N_remain = malloc(K*sizeof(int_t));
            r_acc = malloc(K*sizeof(real_t));
            logp_acc_buf = malloc(K*sizeof(real_t));
            n_cur = malloc(K*sizeof(int_t));
        }
        iterative_cells_core_simd_opt(N, K, reward, log_prob, p_acc, 0, N, 0, logp_acc,
                                       N_remain, r_acc, logp_acc_buf, n_cur);
        if (K > CELLS_OPT_MAX_DEPTH) { free(N_remain); free(r_acc); free(logp_acc_buf); free(n_cur); }
        return;
    }

    const int_t n_threads = threads > N+1 ? N+1 : threads;  // no point over-subscribing n0 in [0, N]

    atomic_int_fast64_t next_n0;
    atomic_init(&next_n0, 0);

    pthread_t* tid = malloc(n_threads*sizeof(pthread_t));
    iterative_cells_simd_opt_args_t* args = malloc(n_threads*sizeof(iterative_cells_simd_opt_args_t));

    for (int_t t=0; t<n_threads; t++) {
        args[t].N = N;
        args[t].K = K;
        args[t].reward = reward;
        args[t].log_prob = log_prob;
        args[t].logp_acc = logp_acc;
        args[t].next_n0 = &next_n0;
        args[t].p_acc = 0;
        pthread_create(&tid[t], NULL, iterative_cells_thread_simd_opt, &args[t]);
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

static void exhaustive_cells_simd_opt(multinomial* mult, multinomial_result* mult_res, options_t* options) {
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
    if (options->threads > 1) iterative_cells_parallel_simd_opt(N, K, reward, log_prob, logp_acc, &p_acc, options->threads);
    else iterative_cells_parallel_simd_opt(N, K, reward, log_prob, logp_acc, &p_acc, 1);
    p_acc = p_acc*EXP(-MAX_EXP);
    if (options->verbose) printf("exhaustive cells SIMD opt = %.8e\n", p_acc);
    mult_res->pval = p_acc;
    mult_res->converged = 1;
    mult_res->terms = -1;

    free(reward);
    free(log_prob);
}
