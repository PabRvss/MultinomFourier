#pragma once

static void create_alias_table_cl(alias_pair_cl* pairs, const small_t bins, const double* pmf) {
    small_t* sm_arr = (small_t*)malloc(bins * sizeof(small_t));
    small_t* lg_arr = (small_t*)malloc(bins * sizeof(small_t));
    double* probs_d = (double*)malloc(bins * sizeof(double));
    small_t n_sm = 0;
    small_t n_lg = 0;

    for (small_t i=0; i < bins; i++) {
        pairs[i].alias = i;

        probs_d[i] = pmf[i] * bins;
        if (probs_d[i] < 1.0) sm_arr[n_sm++] = i;
        else lg_arr[n_lg++] = i;
    }

    while ((n_sm > 0) && (n_lg > 0)) {
        small_t s = sm_arr[--n_sm];
        small_t l = lg_arr[--n_lg];

        pairs[s].alias = l;
        probs_d[l] = probs_d[l] + probs_d[s] - 1.0;
        if (probs_d[l] < 1.0) sm_arr[n_sm++] = l;
        else lg_arr[n_lg++] = l;
    }

    while (n_lg > 0) probs_d[lg_arr[--n_lg]] = 1.0;
    while (n_sm > 0) probs_d[sm_arr[--n_sm]] = 1.0;

    for (small_t i=0; i<bins; i++) {
        if (pairs[i].alias == i) probs_d[i] = 1.0;
        pairs[i].prob = (real_t)(i + probs_d[i]);
    }

    free(sm_arr);
    free(lg_arr);
    free(probs_d);
}

static void create_binomial_table_cl(alias_pair_cl* pairs, const small_t n, double p) {
    const small_t N = n+1;
    double* pmf = (double*)malloc(N*sizeof(double));

    if ((p > 0.0) && (p < 1.0)) {
        small_t index = 0;
        small_t step = 1;
        if (p > 0.500001) {
            index = n;
            step = -1;
            p = 1.0-p;
        }

        double q = 1.0-p;
        double d = 1.0/q;
        double t = ipow(q, n);

        pmf[index] = t;
        for (small_t i=1; i<N; i++) {
            t *= p*d*(N-i)/i;
            index += step;
            pmf[index] = t;
        }
    }
    else if (p <= 0.0) {
        pmf[0] = 1.0;
        for (small_t i=1; i<N; i++) pmf[i] = 0.0;
    }
    else if (p >= 1.0) {
        for (small_t i=0; i<N-1; i++) pmf[i] = 0.0;
        pmf[N-1] = 1.0;
    }
    create_alias_table_cl(pairs, N, pmf);

    free(pmf);
}
