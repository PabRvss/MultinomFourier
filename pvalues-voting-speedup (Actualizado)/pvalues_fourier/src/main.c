#include <string.h>
#include <complex.h>
#include <quadmath.h>
#include <pthread.h>

#include "base.h"
#include "rewards.h"
#include "options.h"

#include "multinomial/multinomial.h"
#include "multinomial/fourier.h"
#include "multinomial/exhaustive_bisection.h"
#include "multinomial/exhaustive_bounds.h"

#ifdef USE_OPENCL
#include "multinomial_cl/base_cl.h"
#include "multinomial_cl/fourier_cl.h"
#endif

static void exhaustive_dispatch(multinomial* mult, multinomial_result* mult_res, options_t* options) {
    if (options->fast_exhaustive) exhaustive_bounds(mult, mult_res, options);
    else exhaustive_bisection(mult, mult_res, options);
}

#include "main_benchmark.h"

// TODO:
// OpenCL global/local memory selection
// OpenCL chunks of series terms
// gamma optimization with OpenCL

void init_ext(int_t fact_max, int_t verbose_ext) {
    if (lgac!=NULL) free(lgac);

    lgac = malloc((fact_max+1)*sizeof(real_t));
    lgac[0] = 0;

    double s = 0;
    double d = 0;
    for (size_t i=1; i <= fact_max; i++) {
        double l = log((double)i);

        double y = l - d;
        double t = s + y;
        d = (t - s) - y;
        s = t;
        lgac[i] = (real_t)s;
    }
}

void finish() {
    free(lgac);
}

void main_int(
        const int_t N, const int_t K,
        const int_t* x0, const real_t* probs,
        multinomial_result* mult_res,
        options_t* options
) {
    const int_t verbose = options->verbose;
    real_t supp = (lgac[N + K - 1] - lgac[N] - lgac[K - 1]) / LOG(10.0);

    if (verbose) {
        printf("p = [");
        for (int_t k = 0; k < K; k++) printf("%.8f ", probs[k]);
        printf("]\n");
        printf("x0 = [");
        for (int_t k = 0; k < K; k++) printf("%lld, ", x0[k]);
        printf("]\n");
        printf("expected = [");
        for (int_t k = 0; k < K; k++) printf("%.3f, ", N*probs[k]);
        printf("]\n");
        printf("N = %lld\n", N);
        printf("K = %lld\n", K);
        printf("log10 support = %.6f\n", supp);
        printf("\n");
    }

    struct timespec t0, t1, t2;
    timespec_get(&t0, TIME_UTC);

    multinomial* mult = get_mult(N, K, probs, options);
    fill_mult1(mult, mult_res, x0, options);
    if (supp >= options->enum_cutoff) fill_mult2(mult, mult_res, x0, options);
    mult_res->p0 = mult->p0;
    timespec_get(&t1, TIME_UTC);
    if (verbose) printf("CPU init time: %.6f seconds\n\n", get_dt(t0, t1));

    if (supp >= options->enum_cutoff) {
        series(mult, mult_res, options);
        timespec_get(&t2, TIME_UTC);
        if (verbose) printf("CPU series time: %.6f seconds\n", get_dt(t1, t2));
    } else {
        // exhaustive_cells_simd(mult, mult_res, options);
        // exhaustive_cells_simd_opt(mult, mult_res, options);
        exhaustive_dispatch(mult, mult_res, options);
        timespec_get(&t2, TIME_UTC);
        if (verbose) printf("exhaustive time: %.6f seconds\n", get_dt(t1, t2));
    }
    if (verbose) printf("CPU total time: %.6f seconds\n\n", get_dt(t0, t2));

#ifdef USE_OPENCL
    load_kernel();

    timespec_get(&t0, TIME_UTC);
    multinomial_cl* mult_cl = malloc(sizeof(multinomial_cl));
    create_multinomial_cl(mult_cl, mult);
    timespec_get(&t1, TIME_UTC);
    printf("CL init time: %.6f seconds\n", get_dt(t0, t1));

    result_cl_t* res = malloc(GROUPS*sizeof(result_cl_t));
    double* res_double = malloc(GROUPS*sizeof(double));
    calculate_multinomial_cl(0, mult_cl, res, res_double);
    free_multinomial_cl(mult_cl);

    free(mult_cl);
    free(res);
    free(res_double);
    free_kernel();

    timespec_get(&t2, TIME_UTC);
    printf("CL series time: %.6f seconds\n", get_dt(t0, t2));
#endif

    free_mult(mult);
}

void main_ext(
        const int_t N, const int_t K,
        const int_t* x0, const real_t* probs,
        double* restrict p_value,
        int_t* restrict converged,
        int_t* restrict terms,
        double* restrict p0
) {
    multinomial_result mult_res;
    options_t options;
    set_options_default(&options);

    main_int(N, K, x0, probs, &mult_res, &options);

    p_value[0] = mult_res.pval;
    converged[0] = mult_res.converged;
    terms[0] = mult_res.terms;
    p0[0] = mult_res.p0;
}

int main() {
    const int_t N_max = 10000;
    const int_t K_max = 100;
    init_ext(N_max+K_max, 1);
    options_t options;
    set_options_default(&options);

    // test instance
    int_t N = 100;
    int_t K = 4;

    real_t* probs = malloc(K*sizeof(real_t));
    int_t* x0 = malloc(K*sizeof(int_t));

    // for (int_t k=0; k<K; k++) probs[k] = pow(k+1, 17.0);
    for (int_t k=0; k<K; k++) probs[k] = pow(k+1, 1.0);
    int_t rem = N; for (int_t k = 0; k < K; k++) {x0[k] = N/K; rem -= x0[k];} x0[K-1] += rem;
    // for (int_t k=0; k<K; k++) probs[k] = 1;
    // for (int_t k = 0; k < K; k++) x0[k] = pow(k+1, 3.0)/50;

    real_t sum = 0; N = 0;
    for (int_t k=0; k<K; k++) {sum += probs[k]; N += x0[k];}
    for (int_t k=0; k<K; k++) probs[k] /= sum;

    // main_fast();

    multinomial_result mult_res;
    main_int(N, K, x0, probs, &mult_res, &options);

    free(probs);
    free(x0);

    // main_benchmark();
    finish();
}
