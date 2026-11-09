#include "base.h"
#include "cpu/pcg_basic.h"
#include "cpu/utils.h"
#include "cpu/stats.h"
#include "cpu/multi_numpy.h"

#ifdef USE_OPENCL
#include "cl/multi_alias_cl.h"
#else
#include "cpu/multi_alias.h"
#endif


inline static void estimate_pvals(small_t n, small_t c, small_t df, output* out, const double* probs, const size* real) {
    double chisq = 0.0;
    double llr = 0.0;
    for (small_t i=0; i < c; i++) {
        if (probs[i] <= 0.0) continue;

        double expected = n*probs[i];
        double diff = (double)real[i] - expected;
        chisq += diff*diff/expected;
        llr += real[i] > 0 ? (double)real[i]*(log(expected/(double)real[i])): 0.0;
    }
    double p_value_chisq = pochisq(chisq, df);
    double p_value_llr = pochisq(-2*llr, df);
    out->p_value_chisq = p_value_chisq;
    out->p_value_llr = p_value_llr;
}

static inline void sample_alias(
        const small_t n, const small_t c,
        const double* probs, const double* log_p, const double* lowp_remain,
        output* out,
        size t_index, size less_p, const double beta_n
    ) {
    double t0m = seconds();

#ifdef USE_OPENCL
    multi_alias_cl mult;
    create_multi_alias_cl(&mult, n, c, probs, beta_n);
#else
    multi_alias mult;
    create_multi_alias(&mult, n, c, probs);
#endif

    if (verbose) {
        double elapsedm = seconds() - t0m;
        printf("alias time: %f\n", elapsedm);
    }

    for (; t_index<trials_n; t_index++) {
        size lower = t_index == 0 ? 0: trials_array[t_index-1];
        size upper = trials_array[t_index];

#if defined(USE_OPENCL)
#ifdef _WIN32
        size step = step_num/c;
        size trials = upper - lower;
        size splits = (trials + step - 1)/step;
        for (size i=0; i<splits; i++) {
            size start = lower + i * step;
            size end = start + step;
            if (end > upper) end = upper;
            less_p += sample_multi_alias_cl(&mult, start, end);
        }
#else
        less_p += sample_multi_alias_cl(&mult, lower, upper);
#endif
#elif defined(PARALLEL_ALIAS)
        less_p += sample_multi_alias_parallel(&mult, beta_n, log_p, lowp_remain, lower, upper);
#else
        arguments_alias args;
        args.mult = &mult;
        args.start = lower;
        args.end = upper;
        args.beta_n = beta_n;
        args.log_p = log_p;
        args.lowp_remain = lowp_remain;
        for (size trial=lower; trial<upper; trial++) less_p += sample_multi_alias(n+1, n, c, &args, trial);
#endif

        double frac = (double)less_p/(double)upper;
        double frac_crit = p_crit_array[t_index];
        if (verbose) printf("\talias %lld:\t p = %.9f\t (crit: %f)\n", upper, frac, frac_crit);

        bool score = wilson_score(frac, upper);

        out->trials = upper;
        out->p_value = frac;
        if ((frac >= frac_crit) || score) break;
    }

#ifdef USE_OPENCL
    free_multi_alias_cl(&mult);
#else
    free_multi_alias(&mult);
#endif
}

static void calculate_samples(const small_t n, const small_t c, const size* real, const double* probs, output* out) {

    out->p_value = NAN;
    out->p_value_chisq = NAN;
    out->p_value_llr = NAN;
    out->trials = 0;

    small_t df = c - 1;
    double* log_p = (double*)malloc(c*sizeof(double));
    for (small_t i=0; i < c; i++) {
        log_p[i] = log(probs[i]);
        if (probs[i] <= 0.0) {
            df -= 1;
            log_p[i] = 0.0;
            if (real[i] > 0) {
                out->p_value = 0.0;
                return;
            }
        }
    }

    if (quick) {
        double s = lgac[n];
        for (small_t i=0; i < c; i++) s += (double)real[i] * log_p[i] - lgac[real[i]];
        if (s >= log(threshold)) return;
    }

    if (estimate) estimate_pvals(n, c, df, out, probs, real);

    double beta_n = 0.0;
    for (small_t i=0; i < c; i++) {
        double first = probs[i] == 0 ? 0.0 : (double)real[i]*log_p[i];
        beta_n += first - lgac[real[i]];
    }

    double* lowp_remain = (double*)malloc(c*sizeof(double));
    double lowp = 0.0;
    for (small_t i= c - 1; i >= 0; i--) {
        if (probs[i] > 0.0) {
            double logpi = log(probs[i]);
            if (logpi < lowp) lowp = logpi;
        }
        lowp_remain[i] = lowp;
    }

    size less_p = 0;

    if (verbose) printf("trials:\n");
    size t_index=0;
    for (; t_index<trials_n; t_index++) {
        size lower = t_index == 0 ? 0: trials_array[t_index-1];
        size upper = trials_array[t_index];
        if (upper > trials_numpy) break;

#ifdef PARALLEL_NUMPY
        less_p += sample_multi_numpy_parallel(n, c, probs, log_p, lowp_remain, beta_n, lower, upper);
#else
        pcg32_random_t rng;
        for (size trial=lower; trial<upper; trial++) less_p += sample_multi_numpy(n, c, probs, beta_n, lowp_remain, trial, log_p, &rng);
#endif

        double frac = (double)less_p/(double)upper;
        double frac_crit = p_crit_array[t_index];
        if (verbose) printf("\tnumpy %lld:\t p = %f\t (crit: %f)\n", upper, frac, frac_crit);

        bool score = wilson_score(frac, upper);

        out->trials = upper;
        out->p_value = frac;
        if ((t_index == trials_n-1) || (frac >= frac_crit) || score) goto Finish;
        else if (upper >= trials_numpy) {
            t_index++;
            break;
        }
    }

    sample_alias(
        n, c,
        probs, log_p, lowp_remain,
        out,
        t_index, less_p, beta_n
    );

    Finish:
    free(lowp_remain);
    free(log_p);
}

EXTERNAL void main_ext(
        pyint n, pyint c,
        const size* real, const double* probs,
        double* p_value, double* p_value_chisq, double* p_value_llr, size* trials
        ) {
    if (verbose) print_instance((small_t)n, (small_t)c, real, probs);

    double t0 = seconds();

    output out;
    calculate_samples((small_t)n, (small_t)c, real, probs, &out);

    if (verbose) {
        double elapsed = seconds() - t0;
        print_results(&out, elapsed);
    }

    p_value[0] = out.p_value;
    p_value_chisq[0] = out.p_value_chisq;
    p_value_llr[0] = out.p_value_llr;
    trials[0] = out.trials;
}

int main() {
//    small_t base[] = {1, 5, 2, 8, 3}; small_t reps = 10;
    small_t base[] = {11, 3, 3, 3, 3}; small_t reps = 13;
//    small_t base[] = {10, 15, 17, 12, 15}; small_t reps = 4;
//    small_t base[] = {120, 180}; small_t reps = 1;
//    small_t base[] = {1, 0, 3, 0, 2, 1, 0, 1, 1, 0, 7, 1, 2, 3, 3, 18, 0, 2, 0, 2, 3, 0, 1, 0, 2, 2, 3, 1, 1, 1, 0, 30, 5, 2, 0, 1, 1, 2, 8, 0, 0, 0, 1, 0, 1, 1, 1, 2, 1, 3, 0, 0, 2, 0, 2, 2, 0, 1, 0, 7, 7}; small_t reps = 1;
//    double probs0[] = {0.013364, 0.005659, 0.005812, 0.008604, 0.022025, 0.004379, 0.007276, 0.009400, 0.002748, 0.015471, 0.059925, 0.007835, 0.006588, 0.023151, 0.034654, 0.014201, 0.064383, 0.024064, 0.002026, 0.026367, 0.016909, 0.011954, 0.030404, 0.004459, 0.009895, 0.007047, 0.009094, 0.007729, 0.017492, 0.003279, 0.006527, 0.172299, 0.009570, 0.018002, 0.008768, 0.004523, 0.004412, 0.003801, 0.048155, 0.004833, 0.006954, 0.001625, 0.002062, 0.003133, 0.000855, 0.030700, 0.023199, 0.013935, 0.014052, 0.015667, 0.009667, 0.013972, 0.007265, 0.008811, 0.001710, 0.002547, 0.001191, 0.001684, 0.002001, 0.041649, 0.040235};

    small_t n_base = sizeof(base) / sizeof(small_t);
    small_t n = 0;
    small_t c = n_base * reps;
    verbose = true;

    double* probs = (double*)malloc(c*sizeof(double));
    size* real = (size*)malloc(c*sizeof(size));
    for (small_t i=0; i < c; i++) probs[i] = 1.0 / c;
//    double p = 0.0; for (small_t i=0; i<c; i++) {probs[i] = probs0[i % n_base]; p += probs[i];} for (small_t i=0; i<c; i++) {probs[i] = probs[i]/p;}
    for (small_t i=0; i < c; i++) {
        real[i] = base[i % n_base];
        n += (small_t)real[i];
    }

    double t0 = seconds();

    init_default();

    if (verbose) {
        double elapsed = seconds() - t0;
        printf("startup time: %f\n", elapsed);
        print_instance(n, c, real, probs);
    }

    t0 = seconds();

    output out;
    calculate_samples(n, c, real, probs, &out);

    if (verbose) {
        double elapsed = seconds() - t0;
        print_results(&out, elapsed);
    }

    free(probs);
    free(real);

    finish();

    return 0;
}
