#include "base.cpp"

static inline double get_dt(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)*1e-9;
}

//
// Demonstrates calling bag_sFFT directly on an observation, with no
// console arguments and no precomputed score -- compare to main() in
// entropy-test.cpp, which instead reads N, K, L, s, Q, pu from argv.
//
int main() {

    const int K = 10;
//    const int K = 2;
//    const int K = 8;

    auto* probs = (double*)malloc(K*sizeof(double));
    auto* x0 = (int*)malloc(K*sizeof(int));

    for (int k=0; k<K; k++) probs[k] = pow(k+1, 0.5);
    for (int k=0; k<K; k++) x0[k] = 1000;

    double sum = 0;
    for (int k=0; k<K; k++) sum += probs[k];
    for (int k=0; k<K; k++) probs[k] /= sum;

    int N = 0;
    for (int k=0; k<K; k++) N += x0[k];

    int Q = 1 << 17;
    printf("Q = %d\n", Q);

    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);
    bag_sFFT_result_t result = run_bag_sFFT_obs(N, K, 1, Q, probs, x0);
    timespec_get(&t1, TIME_UTC);

    double log_pval = (result.log_max_pval+result.log_min_pval)/2;

    cout << "Score = "
         << setprecision(10) << result.score << endl;
    cout << "P = " << setprecision(10) << log_pval
         << " (" << exp(log_pval) << ")" << endl;
    cout << "Max = " << setprecision(10) << result.log_max_pval
         << " (" << exp(result.log_max_pval) << ")" << endl;
    cout << "Min = " << setprecision(10) << result.log_min_pval
         << " (" << exp(result.log_min_pval) << ")" << endl;
    cout << "err Max = " << setprecision(10) << result.log_max_pval_err_bnd
         << " (" << exp(result.log_max_pval_err_bnd) << ")" << endl;
    cout << "err Min = " << setprecision(10) << result.log_min_pval_err_bnd
         << " (" << exp(result.log_min_pval_err_bnd) << ")" << endl;

    printf("time: %.6f seconds\n\n", get_dt(t0, t1));

    return 0;
}
