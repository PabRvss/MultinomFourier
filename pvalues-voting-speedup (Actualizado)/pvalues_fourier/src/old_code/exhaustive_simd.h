#pragma once
#include <stdatomic.h>
#include "../sse_mathfun_generic/mathfun_generic.h"

// SIMD variant of iterative_core (exhaustive.h): identical stack-based
// composition enumeration, except the loop over the second-to-last
// category's count (the "k == K-2" leaf-producing level, where the last
// category K-1 is always whatever remains) is batched STRIDE values at a
// time, matching how eval_fourier_thread/eval_gamma_thread use vec_t:
// reward_power/reward_logp involve a pow() call and a lgac[] table lookup,
// so they're filled into vec_t lanes with a per-lane scalar loop (like
// eval_fourier_thread fills cos/sin per lane), while the additions that
// combine each lane's reward with the (lane-invariant) inherited prefix run
// as real vec_t SIMD ops (like eval_gamma_thread's vec_t exponent build).
// The final r0 compare and EXP accumulation stay a per-lane scalar loop too
// (like eval_gamma_thread's `res[s] += EXP(exponent[s])`), since there's no
// masked-add primitive in this codebase's SIMD macros.
//
// Levels above k == K-2 keep the plain scalar traversal from iterative_core
// unchanged: their subtree sizes vary per branch, so there's no uniform,
// fixed-shape loop to batch there the way Fourier batches consecutive n or
// gamma does.
//
// reward_power(N, k, log_pk, xk, lambda) (rewards.h) branches on lambda into
// one of three statistics (pmf / log-likelihood-ratio / general power
// divergence), but lambda is a single fixed value for an entire
// exhaustive_simd() call -- it never varies across k or xk -- so which
// branch applies can be decided once per leaf-loop invocation instead of
// once per xk. Only the general-power branch (pow_vec, sse_mathfun_generic)
// is worth batching across STRIDE lanes this way: libm's pow() is itself
// roughly an exp(log()) internally, so replacing STRIDE scalar pow() calls
// with one pow_vec() call has real headroom to win. The pmf/llr branches
// are cheap enough already (llr is a single log() call with an xk==0 early
// out, pmf is just arithmetic + a lgac[] lookup) that going through
// log_vec's masking/blending overhead measured out as a net loss for them,
// so they keep the original per-lane reward_power() calls below.
typedef enum { REWARD_CASE_PMF, REWARD_CASE_LLR, REWARD_CASE_POWER } reward_case_t;

#ifdef USE_SIMD
static inline reward_case_t reward_power_case(const real_t lambda) {
    if (real_isnan(lambda)) return REWARD_CASE_PMF;
    if (ABS(lambda) <= EPS) return REWARD_CASE_LLR;
    return REWARD_CASE_POWER;
}

// Vectorized general-power reward_power() across STRIDE lanes of xk
// (xk_vec), for the single category (log_pk fixed) and single lambda a
// whole leaf-loop invocation shares.
static inline vec_t reward_power_vec(const int_t N, const real_t log_pk, const vec_t xk_vec, const real_t lambda) {
    const vec_t one_vec = VEC_ONE;
    const vec_mask_t zero_mask = VEC_CMP_MASK(xk_vec, VEC_ZERO, SIMDE_CMP_EQ_OQ);  // xk == 0 -> reward 0

    const real_t inv_Np = 1.0/(N*EXP(log_pk));
    const vec_t base_vec = xk_vec*(inv_Np*one_vec);
    const vec_t powed_vec = pow_vec(base_vec, lambda*one_vec);
    const real_t coeff = -1.0/(lambda*(lambda+1));
    const vec_t r_vec = (coeff*one_vec)*xk_vec*(powed_vec - one_vec);
    return VEC_BLEND_REAL(zero_mask, r_vec, VEC_ZERO);
}
#endif

static void iterative_core_simd(
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
            const real_t log_p = log_probs[k];
            const real_t log_p_last = log_probs[K-1];
            const real_t r_acc_d = r_acc[d];
            const real_t logp_acc_d = logp_acc[d];
            const int_t N_remain_d = N_remain[d];
            real_t p_leaf = 0;

#ifdef USE_SIMD
            const reward_case_t case_ = reward_power_case(lambda);
            const vec_t r_acc_vec = r_acc_d*VEC_ONE;
            const vec_t logp_acc_vec = logp_acc_d*VEC_ONE;

            for (int_t n = n_cur[d]; n <= N_remain_d; n += STRIDE) {
                vec_t reward_vec, dp_vec, reward_last_vec, dp_last_vec;
                int_t valid[STRIDE];

                if (case_ == REWARD_CASE_POWER) {
                    vec_t xk_vec, xk_last_vec;
                    for (int s=0; s<STRIDE; s++) {
                        const int_t n_s = n + s;
                        valid[s] = n_s <= N_remain_d;
                        const int_t n_use = valid[s] ? n_s : N_remain_d;  // clamp padding lanes to a safe xk
                        const int_t N_next = N_remain_d - n_use;

                        xk_vec[s] = (real_t)n_use;
                        dp_vec[s] = reward_logp(log_p, n_use);
                        xk_last_vec[s] = (real_t)N_next;
                        dp_last_vec[s] = reward_logp(log_p_last, N_next);
                    }
                    reward_vec = reward_power_vec(N, log_p, xk_vec, lambda);
                    reward_last_vec = reward_power_vec(N, log_p_last, xk_last_vec, lambda);
                } else {
                    for (int s=0; s<STRIDE; s++) {
                        const int_t n_s = n + s;
                        valid[s] = n_s <= N_remain_d;
                        const int_t n_use = valid[s] ? n_s : N_remain_d;  // clamp padding lanes to a safe xk
                        const int_t N_next = N_remain_d - n_use;

                        reward_vec[s] = reward_power(N, k, log_p, n_use, lambda);
                        dp_vec[s] = reward_logp(log_p, n_use);
                        reward_last_vec[s] = reward_power(N, K-1, log_p_last, N_next, lambda);
                        dp_last_vec[s] = reward_logp(log_p_last, N_next);
                    }
                }

                const vec_t r_final_vec = r_acc_vec + reward_vec + reward_last_vec;
                const vec_t logp_final_vec = logp_acc_vec + dp_vec + dp_last_vec;

                for (int s=0; s<STRIDE; s++) {
                    if (valid[s] && r_final_vec[s] <= r0) p_leaf += EXP(MAX_EXP + logp_final_vec[s]);
                }
            }
#else
            for (int_t n = n_cur[d]; n <= N_remain_d; n++) {
                const int_t N_next = N_remain_d - n;
                const real_t r_final = r_acc_d + reward_power(N, k, log_p, n, lambda)
                                                + reward_power(N, K-1, log_p_last, N_next, lambda);
                const real_t logp_final = logp_acc_d + reward_logp(log_p, n)
                                                      + reward_logp(log_p_last, N_next);
                if (r_final <= r0) p_leaf += EXP(MAX_EXP + logp_final);
            }
#endif
            // Folded as one local sum then added once, instead of accumulating each
            // term into *p_acc individually: mathematically identical to
            // iterative_core, but not guaranteed bit-identical since floating-point
            // addition isn't associative (same harmless class of reordering as
            // iterative_parallel's thread-sum, just from block-summing here instead).
            *p_acc = *p_acc + p_leaf;

            d--;
            if (d >= 0) n_cur[d]++;
            continue;
        }

        const int_t n = n_cur[d];
        const real_t log_p = log_probs[k];
        const real_t r_next = r_acc[d] + reward_power(N, k, log_p, n, lambda);
        const real_t logp_next = logp_acc[d] + reward_logp(log_p, n);
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

static void iterative_simd(
        const real_t lambda,
        const int_t N,
        const int_t K,
        const real_t* restrict log_probs,
        real_t* restrict p_acc,
        const real_t r0,
        const real_t logp_acc0
) {
    iterative_core_simd(lambda, N, K, log_probs, p_acc, r0, 0, N, 0, logp_acc0);
}

typedef struct {
    real_t lambda;
    int_t N, K;
    const real_t* restrict log_probs;
    real_t r0;
    real_t logp_acc0;
    atomic_int_fast64_t* restrict next_n0;  // shared work queue: next category-0 count to claim
    real_t p_acc;  // thread-local partial sum, written once before the thread exits
} iterative_worker_simd_args_t;

// Same dynamic work-queue strategy as iterative_parallel (exhaustive.h): the
// per-n0 sub-problem sizes are wildly uneven, so threads claim the next
// unclaimed n0 from a shared atomic counter rather than owning a static
// range, and each claimed n0's sub-enumeration runs through the
// SIMD-batched iterative_core_simd.
static void* iterative_worker_simd(void* args_void) {
    iterative_worker_simd_args_t* args = (iterative_worker_simd_args_t*)args_void;
    const int_t N = args->N;
    const real_t lambda = args->lambda;
    const real_t* restrict log_probs = args->log_probs;

    real_t p_acc = 0;
    int_t n0;
    while ((n0 = atomic_fetch_add_explicit(args->next_n0, 1, memory_order_relaxed)) <= N) {
        const real_t log_p0 = log_probs[0];
        const real_t r1 = reward_power(N, 0, log_p0, n0, lambda);
        const real_t logp1 = args->logp_acc0 + reward_logp(log_p0, n0);
        iterative_core_simd(lambda, N, args->K, log_probs, &p_acc, args->r0, 1, N-n0, r1, logp1);
    }

    args->p_acc = p_acc;
    return NULL;
}

static void iterative_parallel_simd(
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
        iterative_core_simd(lambda, N, K, log_probs, p_acc, r0, 0, N, 0, logp_acc0);
        return;
    }

    const int_t n_threads = threads > N+1 ? N+1 : threads;  // no point over-subscribing n0 in [0, N]

    atomic_int_fast64_t next_n0;
    atomic_init(&next_n0, 0);

    pthread_t* tid = malloc(n_threads*sizeof(pthread_t));
    iterative_worker_simd_args_t* args = malloc(n_threads*sizeof(iterative_worker_simd_args_t));

    for (int_t t=0; t<n_threads; t++) {
        args[t].lambda = lambda;
        args[t].N = N;
        args[t].K = K;
        args[t].log_probs = log_probs;
        args[t].r0 = r0;
        args[t].logp_acc0 = logp_acc0;
        args[t].next_n0 = &next_n0;
        args[t].p_acc = 0;
        pthread_create(&tid[t], NULL, iterative_worker_simd, &args[t]);
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

static void exhaustive_simd(multinomial* mult, multinomial_result* mult_res, options_t* options) {
    const real_t lambda = options->lambda;
    const int_t N = mult->N;
    const int_t K = mult->K;
    const real_t r0 = mult->r0;
    real_t p_acc = 0;
    const real_t logp_acc = lgac[N];
    if (options->threads > 1) iterative_parallel_simd(lambda, N, K, mult->log_probs, &p_acc, r0, logp_acc, options->threads);
    else iterative_core_simd(lambda, N, K, mult->log_probs, &p_acc, r0, 0, N, 0, logp_acc);
    p_acc = p_acc*EXP(-MAX_EXP);
    if (options->verbose) printf("exhaustive SIMD = %.8e\n", p_acc);
    mult_res->pval = p_acc;
    mult_res->converged = 1;
    mult_res->terms = -1;
}
