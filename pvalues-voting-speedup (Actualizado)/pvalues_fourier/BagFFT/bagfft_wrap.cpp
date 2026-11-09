#include <cmath>
#include <cstdlib>

#include "base.cpp"
#include "bagfft.h"

extern "C" double bagfft_llr_pvalue(int N, int K, const int* x0, const double* probs, int Q) {
    if (Q <= 0) Q = 1 << 16;

    auto* pu = (double*)malloc(K*sizeof(double));
    auto* counts = (int*)malloc(K*sizeof(int));
    for (int k = 0; k < K; k++) {
        pu[k] = probs[k];
        counts[k] = x0[k];
    }

    bag_sFFT_result_t result = run_bag_sFFT_obs(N, K, 1, Q, pu, counts);

    free(pu);
    free(counts);

    double log_pval = (result.log_min_pval + result.log_max_pval) / 2;
    return exp(log_pval);
}
