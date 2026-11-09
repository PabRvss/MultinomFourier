#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// N: total count (sum of x0). K: number of categories.
// x0: observed counts, length K. probs: hypothesized probabilities, length K (should sum to 1).
// Q: FFT grid resolution (a power of two); pass 0 to use this wrapper's default (1<<16 = 65536).
// Larger Q trades runtime for precision; smaller Q degrades smoothly rather than failing outright.
//
// Returns the estimated p-value.
double bagfft_llr_pvalue(int N, int K, const int* x0, const double* probs, int Q);

#ifdef __cplusplus
}
#endif
