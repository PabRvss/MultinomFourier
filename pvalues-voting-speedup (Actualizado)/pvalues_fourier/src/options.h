#pragma once

typedef enum {
    FFT_NONE,
    FFT_REAL,
} fft_precision_t;

typedef struct {
    real_t lambda;

    real_t eps_rel;
    int_t max_iter;
    real_t undersampling;
    int_t B;

    real_t eps_gamma;
    real_t enum_cutoff;

    int_t error_bound;  // compute truncation diagnostics in series() (see multinomial/error_bound.h)
    // Gamma search (see multinomial/gamma_search.h): 0 = exact grid refinement,
    // 1 = Newton on the saddlepoint surrogate with analytic derivatives.
    int_t gamma_newton;
    // Series evaluator: 1 = Poissonised Cauchy-integral evaluator (see multinomial/fourier_poisson.h),
    // overriding fft_precision/use_fft_precompute for the series terms.
    int_t poisson;
    // Statistic interval (see multinomial/interval.h): 1 = the O(K) closed-form min/max for the
    // power-divergence family (any real lambda, LLR included), or the O(K log K) GreedyModeFind-
    // based exact min/max for the pmf statistic (lambda = NaN); either way replaces fill_interval's
    // O(K*N^2) dynamic program.
    int_t fast_interval;
    // Exhaustive method, used when supp < enum_cutoff (see multinomial/exhaustive_bisection.h and
    // multinomial/exhaustive_bounds.h): 0 = bisection on the reward's sign change in the last two
    // categories only, 1 = per-node dynamic-programming bounds that prune the whole tree.
    int_t fast_exhaustive;
    // Convergence acceleration of the partial sums (see multinomial/extrapolate.h):
    // 0 off, 1 report the extrapolated value, 2 also stop on its error estimate.
    // Ignored when average_window > 0, which keeps the trailing-window average instead.
    int_t extrapolate;

    fft_precision_t gamma_precision;
    fft_precision_t fft_precision;
    int_t use_fft_precompute;

    int_t threads;
    int_t vecs_per_thread;

    int_t verbose;
    int_t print_freq;
} options_t;

static void set_options_default(options_t* options) {
    // options->lambda = NAN;  // for pmf statistic
    options->lambda = 0.0;  // 0 for LLR, 1 for chi-squared

    options->eps_rel = 1e-6;
    options->max_iter = 100000;
    options->undersampling = 1.0;
    options->B = 50;

    options->eps_gamma = 1e-2;
    // options->enum_cutoff = INFINITY;
    options->enum_cutoff = 0.0;

    options->error_bound = 1;
    options->gamma_newton = 1;
    options->poisson = 1;
    options->extrapolate = 2;
    options->fast_interval = 1;
    options->fast_exhaustive = 1;

    options->gamma_precision = FFT_REAL;
    options->fft_precision = FFT_REAL;
    options->use_fft_precompute = 1;

    options->threads = 8;
    options->vecs_per_thread = 1;

    options->verbose = 1;
    options->print_freq = 100;
}

static void set_options(
    options_t* options,

    real_t eps_rel,
    int_t max_iter,
    real_t undersampling,

    real_t eps_gamma,
    real_t enum_cutoff,

    int_t threads,
    int_t vecs_per_thread,

    int_t verbose,
    int_t print_freq
) {
    set_options_default(options);

    options->lambda = 0;  // TODO: complete

    options->eps_rel = eps_rel;
    options->max_iter = max_iter;
    options->undersampling = undersampling;

    options->eps_gamma = eps_gamma;
    options->enum_cutoff = enum_cutoff;

    options->threads = threads;
    options->vecs_per_thread = vecs_per_thread;

    options->verbose = verbose;
    options->print_freq = print_freq;
}
