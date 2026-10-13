#pragma once

static inline real_t reward_logp(const real_t log_pk, const int_t xk) {  // probability mass statistic
    return xk*log_pk - lgac[xk];
}

static inline real_t reward_llr(const int_t N, const real_t log_pk, const int_t xk, const real_t lambda) {  // log-likelihood ratio
    if (xk == 0) return 0;
    const real_t log_base = LOG((real_t)xk/N) - log_pk;
//    return -xk*log_base*(1 + lambda*log_base/2);
    return -xk*log_base;
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-fast-math"), noinline))
#elif defined(__clang__)
__attribute__((noinline))
#endif
static real_t reward_power(const int_t N, const int_t k, const real_t log_pk, const int_t xk, const real_t lambda) {  // Power divergence statistic (general lambda)
    // Special cases:
    if (real_isnan(lambda)) return reward_logp(log_pk, xk);  // undefined lambda, use probability mass statistic
    if (ABS(lambda) <= EPS) return reward_llr(N, log_pk, xk, lambda);  // log-likelihood ratio is the limit lambda -> 0

    if (xk == 0) return 0;
    const real_t base = xk/(N*EXP(log_pk));
    return -1/(lambda*(lambda+1))*xk*(POW(base, lambda) - 1);
}
