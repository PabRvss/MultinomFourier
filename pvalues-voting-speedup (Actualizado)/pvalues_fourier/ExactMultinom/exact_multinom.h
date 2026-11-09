#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// K: number of categories.
// x0: observed counts, length K.
// probs: hypothesized probabilities, length K (should sum to 1; the original implementation
//        recommends ordering categories from largest to smallest probability for best performance).
// theta: p-values below this threshold are only upper-bounded, not computed exactly (matches the
//        original implementation's own early-termination criterion). Pass 0 to use its own
//        default (1e-4).
//
// Returns the p-value: P(LLR statistic under the null >= the LLR statistic observed at x0).
double exact_multinom_llr_pvalue(int K, const int* x0, const double* probs, double theta);

// Same algorithm, stopping condition, and return convention as exact_multinom_llr_pvalue (same
// upper-bound-on-early-termination behavior, same catastrophic-cancellation susceptibility for
// very small theta) -- multinom_llr_reduced.cpp just drops the probability-mass statistic
// (stat_prob/newProbs_prob/sumProbs_prob) that exact_multinom_llr_pvalue computes but never
// returns. Exists to isolate how much of the original's runtime that unused tracking costs. Since
// the original's stopping condition continues as long as either the prob or the llr mass hasn't
// reached 1-theta, this can very occasionally stop one shell earlier than the original and return
// a very slightly different (still theta-bounded) p-value.
double exact_multinom_llr_pvalue_reduced(int K, const int* x0, const double* probs, double theta);

// Same computation as exact_multinom_llr_pvalue, using multinom_llr_modified.cpp instead of the
// original multinom_llr.cpp: accumulates the matching-tail mass directly, then adds back whatever
// probability mass is still unclassified when the loop stops -- same conservative "don't
// underestimate" bias as the original's 1-minus-complement, but computed as a well-conditioned
// small+small addition instead of a near-1 subtraction, so it can't go negative the way the
// original can for very small theta. The loop stops once no more than theta worth of mass remains
// unclassified, so the returned value is *normally* a provable upper bound within theta of the
// true p-value. Pass 0 for theta to use the same default (1e-4).
//
// reliable: optional (pass NULL if not needed). Set to 0 when theta was set below this specific
// instance's own floating-point noise floor (observed around 1e-12 to 1e-13 for an N=500
// instance, smaller instances resolve further) -- in that case the returned value is just
// whatever tail mass had been explicitly found so far, not a value with the requested theta
// guarantee: the true p-value could be larger by an unknown amount. A microscopic overshoot
// smaller than theta itself is ordinary termination rounding, not this failure, and still
// reports 1. Set to 1 when the theta target was reached normally, which carries the usual
// upper-bound guarantee.
double exact_multinom_llr_pvalue_modified(int K, const int* x0, const double* probs, double theta, int* reliable);

#ifdef __cplusplus
}
#endif
