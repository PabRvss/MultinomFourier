struct bag_sFFT_result_t {

    double score;                  // LLR score computed from the observation

    double log_max_pval;           // Upper bound of log of the p-value
    double log_min_pval;           // Lower bound of log of the p-value

    double log_max_pval_err_bnd;   // Upper bound accounting for pmf error
    double log_min_pval_err_bnd;   // Lower bound accounting for pmf error
};

double compute_llr_score(int N, int K, int L, const double* pu, const int* counts) {
    double score = 0.0;
    for(int l = 0; l < L; l++) {
        for(int k = 0; k < K; k++) score += counts[l*K + k] * log((double) counts[k] / (N*pu[k]));
    }
    return score;
}

bag_sFFT_result_t run_bag_sFFT_obs(int N, int K, int L, int Q, double* pu, int* flat_counts) {
    double s = compute_llr_score(N, K, L, pu, flat_counts);

    double step; double theta; double log_mgf; double* log_err_bnd;
    double* log_pmfL = bag_sFFT(N, K, L, s, pu, Q, step, theta, log_mgf, log_err_bnd);

    bag_sFFT_result_t result;
    result.score = s;

    compute_theta_pvalues(log_pmfL, step, L*Q, s, L*K, result.log_min_pval, result.log_max_pval, theta);
    compute_theta_err_bnds(log_err_bnd, step, L*Q, s, L*K, result.log_min_pval_err_bnd, result.log_max_pval_err_bnd, theta);

    delete[](log_pmfL);
    delete[](log_err_bnd);

    return result;
}
