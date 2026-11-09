#pragma once

#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/select.h>

#include "../ExactMultinom/exact_multinom.h"
#include "../BagFFT/bagfft.h"

#define BENCHMARK_TIMEOUT_SECONDS 1800.0
#define RUN_FOURIER 1
#define RUN_BAGFFT 1
#define RUN_EXHAUSTIVE 0
#define RUN_EXACT_MULTINOM 0
#define SUPPORT_CUTOFF 13.5
// #define SUPPORT_CUTOFF 15.5

static const int Q_values[] = {1<<14, 1<<17};

static const int_t N_values[] = {100, 1000};  // fig1
// static const int_t N_values[] = {50, 100, 200, 500, 1000};  // tab1
// static const int_t N_values[] = {100, 500, 1000, 5000, 10000};  // tab2

static const int_t K_values[] = {3, 5};  // fig1
// static const int_t K_values[] = {3, 5, 7, 9};  // tab1
// static const int_t K_values[] = {5, 10, 20, 50};  // tab2

// static const real_t lambda_values[] = {NAN, 0.0, 1.0};  // fig1
static const real_t lambda_values[] = {0.0};  // tab1, tab2

static const real_t k_power_values[] = {0.5, 1.0};

// static const int_t L_values[] = {
//     10, 13, 16, 20, 25, 32, 40, 50, 63, 79,
//     100, 126, 158, 200, 251, 316, 398, 501, 631, 794,
//     1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943,
//     10000, 12589, 15849, 19953, 25119, 31623, 39811, 50119, 63096, 79433,
//     100000, 125893, 158489, 199526, 251189, 316228, 398107, 501187, 630957, 794328,
//     1000000
// };  // fig1
static const int_t L_values[] = {100000};  // tab1, tab2

static const real_t eps_rel_values[] = {1e-3, 1e-6, 0.0};  // fig1
// static const real_t eps_rel_values[] = {1e-3, 1e-6};  // tab1, tab2

typedef struct {
    quad_t pval;
    double time;
    int converged;
    int timed_out;

    int L_stop;
    int_t Q;
    double time_gamma;
    double err_rel_delta;
    double err_rel_neff;
} benchmark_result_t;

static const benchmark_result_t no_result = {
    .pval = NAN, .time = NAN, .converged = 0, .timed_out = 0,
    .L_stop = -1, .time_gamma = NAN, .Q = -1,
    .err_rel_delta = NAN, .err_rel_neff = NAN
};

typedef struct {
    int_t idx;

    int_t N;
    int_t K;
    real_t k_power;
    real_t log_support;

    options_t options_base;
    options_t* options_fourier;
    double theta;

    int_t results_fourier_len;

    benchmark_result_t result_exhaustive;
    benchmark_result_t result_exact_multinom;
    benchmark_result_t* results_bagfft;
    benchmark_result_t* results_fourier;
} benchmark_instance_t;

static void init_benchmark_instance(benchmark_instance_t* instance) {
    const int_t eps_len = sizeof(eps_rel_values)/sizeof(eps_rel_values[0]);
    const int_t L_len = sizeof(L_values)/sizeof(L_values[0]);
    instance->results_fourier_len = eps_len*L_len;
    instance->theta = 1e-12;
    const int_t Q_len = sizeof(Q_values)/sizeof(Q_values[0]);

    instance->result_exhaustive = no_result;
    instance->result_exact_multinom = no_result;
    instance->options_fourier = malloc(instance->results_fourier_len*sizeof(options_t));
    instance->results_fourier = malloc(instance->results_fourier_len*sizeof(benchmark_result_t));
    instance->results_bagfft = malloc(Q_len*sizeof(benchmark_result_t));

    int_t idx = 0;
    for (int i1=0; i1<eps_len; i1++) {
    for (int i2=0; i2<L_len; i2++) {
        instance->options_fourier[idx] = instance->options_base;
        instance->options_fourier[idx].eps_rel = eps_rel_values[i1];
        instance->options_fourier[idx].max_iter = L_values[i2];

        instance->results_fourier[idx] = no_result;
        idx++;
    }
    }
    for (int i=0; i<Q_len; i++) instance->results_bagfft[i] = no_result;
}

static void free_benchmark_instance(benchmark_instance_t* instance) {
    free(instance->options_fourier);
    free(instance->results_fourier);
    free(instance->results_bagfft);
}

static void fill_log_support(benchmark_instance_t* instance) {
    instance->log_support = lgac[instance->N + instance->K-1] - lgac[instance->N] - lgac[instance->K-1];
}

static inline int idx_cmp(const void *a, const void *b) {
    benchmark_instance_t x = *(const benchmark_instance_t*)a;
    benchmark_instance_t y = *(const benchmark_instance_t*)b;
    return (int)(x.idx - y.idx);
}
static inline int support_cmp(const void *a, const void *b) {
    benchmark_instance_t x = *(const benchmark_instance_t*)a;
    benchmark_instance_t y = *(const benchmark_instance_t*)b;

    const real_t diff = round((x.log_support - y.log_support)*1e6);
    if (diff < 0) return -1;
    if (diff > 0) return 1;
    return idx_cmp(a, b);
}

typedef benchmark_result_t (*timed_fn_t)(void* arg);

static benchmark_result_t run_with_timeout(const timed_fn_t fn, void* arg, const double timeout_seconds) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int pipefd[2];
    if (pipe(pipefd) != 0) return no_result;

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return no_result;
    }

    if (pid == 0) {
        close(pipefd[0]);
        benchmark_result_t result = fn(arg);
        ssize_t written = write(pipefd[1], &result, sizeof(result));
        (void)written;
        close(pipefd[1]);
        _exit(0);
    }

    // Parent
    close(pipefd[1]);

    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_sec += (time_t)timeout_seconds;
    deadline.tv_nsec += (long)((timeout_seconds - (time_t)timeout_seconds) * 1e9);
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000000000L;
    }

    benchmark_result_t result = no_result;
    int timed_out = 0;
    while (1) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double remaining = (double)(deadline.tv_sec - now.tv_sec) + (double)(deadline.tv_nsec - now.tv_nsec) / 1e9;
        if (remaining <= 0) {
            timed_out = 1;
            break;
        }

        struct timeval tv;
        tv.tv_sec = (time_t)remaining;
        tv.tv_usec = (suseconds_t)((remaining - (double)tv.tv_sec) * 1e6);

        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(pipefd[0], &readfds);
        int rv = select(pipefd[0] + 1, &readfds, NULL, NULL, &tv);

        if (rv > 0) {
            ssize_t n = read(pipefd[0], &result, sizeof(result));
            if (n != (ssize_t)sizeof(result)) result = no_result;
            break;
        } else if (rv < 0 && errno == EINTR) {
            continue; // recompute remaining time and retry the select
        } else {
            timed_out = 1;
            break;
        }
    }

    if (timed_out) {
        kill(pid, SIGKILL);
        result = no_result;
        result.converged = 0;
    }

    int status;
    waitpid(pid, &status, 0);
    close(pipefd[0]);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    result.time = get_dt(t0, t1);
    result.timed_out = timed_out;

    return result;
}

typedef struct {
    int_t N;
    int_t K;
    int_t* x0;
    real_t* p;
    options_t* options;
} fourier_args_t;

static benchmark_result_t fourier_adapter(void* arg) {
    fourier_args_t* a = arg;
    multinomial* mult = get_mult(a->N, a->K, a->p, a->options);
    multinomial_result mult_res;
    fill_mult1(mult, &mult_res, a->x0, a->options);
    fill_mult2(mult, &mult_res, a->x0, a->options);
    series(mult, &mult_res, a->options);
    free_mult(mult);

    benchmark_result_t res = no_result;
    res.pval = mult_res.pval;
    res.converged = (int)mult_res.converged;
    res.L_stop = (int)mult_res.terms;
    res.Q = mult_res.Q;
    res.time_gamma = mult_res.eval_time;
    res.err_rel_delta = mult_res.err_rel_ext;
    res.err_rel_neff = mult_res.err_rel_est;

    return res;
}

static benchmark_result_t exhaustive_adapter(void* arg) {
    fourier_args_t* a = arg;
    multinomial* mult = get_mult(a->N, a->K, a->p, a->options);
    multinomial_result mult_res;
    fill_mult1(mult, &mult_res, a->x0, a->options);
    exhaustive_dispatch(mult, &mult_res, a->options);
    free_mult(mult);

    benchmark_result_t res = no_result;
    res.pval = mult_res.pval;
    res.converged = 1;
    return res;
}

typedef struct {
    int K;
    const int* x0_i;
    const double* probs_d;
    double theta;
} exact_multinom_args_t;

static benchmark_result_t exact_multinom_adapter(void* arg) {
    exact_multinom_args_t* a = arg;

    benchmark_result_t res = no_result;
    res.pval = exact_multinom_llr_pvalue(a->K, a->x0_i, a->probs_d, a->theta);
    res.converged = 1;
    return res;
}

typedef struct {
    int N, K;
    const int* x0_i;
    const double* probs_d;
    int Q;
} bagfft_args_t;

static benchmark_result_t bagfft_adapter(void* arg) {
    bagfft_args_t* a = arg;

    benchmark_result_t res = no_result;
    res.pval = bagfft_llr_pvalue(a->N, a->K, a->x0_i, a->probs_d, a->Q);
    res.converged = 1;
    return res;
}

static void sort_instance_desc(const int_t K, real_t* restrict probs, int_t* restrict x0) {
    for (int_t i = 1; i < K; i++) {
        const real_t p = probs[i];
        const int_t x = x0[i];
        int_t j = i - 1;
        while (j >= 0 && probs[j] < p) {
            probs[j+1] = probs[j];
            x0[j+1] = x0[j];
            j--;
        }
        probs[j+1] = p;
        x0[j+1] = x;
    }
}

static void run_benchmark_instance(benchmark_instance_t* instance, int run_ffts, int run_exhaustive, int run_exact_multinom) {
    const int_t N = instance->N;
    const int_t K = instance->K;

    real_t* restrict probs = malloc(K*sizeof(real_t));
    int_t* restrict x0 = malloc(K*sizeof(int_t));

    for (int_t k=0; k<K; k++) probs[k] = pow(k+1, instance->k_power);

    real_t sum = 0;
    for (int_t k=0; k<K; k++) sum += probs[k];
    for (int_t k=0; k<K; k++) probs[k] /= sum;

    int_t rem = N;
    for (int_t k = 0; k < K; k++) { x0[k] = N/K; rem -= x0[k]; }
    x0[K-1] += rem;

    int* restrict x0_i = malloc(K*sizeof(int));
    double* restrict probs_d = malloc(K*sizeof(double));
    for (int_t k = 0; k < K; k++) {
        x0_i[k] = (int)x0[k];
        probs_d[k] = (double)probs[k];
    }

    sort_instance_desc(K, probs, x0);

    if (run_ffts) {
        if (RUN_FOURIER) {
            for (int i=0; i<instance->results_fourier_len; i++) {
                fourier_args_t fourier_args = {.N=N, .K=K, .x0=x0, .p=probs, .options=&instance->options_fourier[i]};
                instance->results_fourier[i] = run_with_timeout(fourier_adapter, &fourier_args, BENCHMARK_TIMEOUT_SECONDS);
            }
        }

        if (RUN_BAGFFT) {
            for (int i=0; i<sizeof(Q_values)/sizeof(Q_values[0]); i++) {
                bagfft_args_t bagfft_args = {.N=(int)N, .K=(int)K, .x0_i=x0_i, .probs_d=probs_d, .Q=Q_values[i]};
                instance->results_bagfft[i] = run_with_timeout(bagfft_adapter, &bagfft_args, BENCHMARK_TIMEOUT_SECONDS);
            }
        }
    }

    if (RUN_EXACT_MULTINOM) {
        if (run_exact_multinom) {
            exact_multinom_args_t exact_multinom_args = { (int)K, x0_i, probs_d, instance->theta };
            instance->result_exact_multinom = run_with_timeout(exact_multinom_adapter, &exact_multinom_args, BENCHMARK_TIMEOUT_SECONDS);
        }
    }

    if (RUN_EXHAUSTIVE) {
        if (run_exhaustive) {
            fourier_args_t fourier_args = {.N=N, .K=K, .x0=x0, .p=probs, .options=&instance->options_base};
            instance->result_exhaustive = run_with_timeout(exhaustive_adapter, &fourier_args, BENCHMARK_TIMEOUT_SECONDS);
        }
    }

    free(probs);
    free(x0);
    free(x0_i);
    free(probs_d);
}

static void print_header() {
    printf("%8s %4s %4s %3s | %14s %10s %6s | %14s %10s | %14s %10s | %14s %10s\n",
           "N", "K", "k", "l", "Fourier p", "time (s)", "terms", "BagFFT p", "time (s)", "ExactMultinom p", "time (s)", "Exhaustive p", "time (s)");
}
static void print_instance(const benchmark_instance_t* instance) {
    char fourier_buf[64], bagfft_buf[64], exact_multinom_buf[64], exhaustive_buf[64];
    quadmath_snprintf(fourier_buf, sizeof(fourier_buf), "%14.6Qe", instance->results_fourier[0].pval);
    quadmath_snprintf(bagfft_buf, sizeof(bagfft_buf), "%14.6Qe", instance->results_bagfft[0].pval);
    quadmath_snprintf(exact_multinom_buf, sizeof(exact_multinom_buf), "%14.6Qe", instance->result_exact_multinom.pval);
    quadmath_snprintf(exhaustive_buf, sizeof(exhaustive_buf), "%14.6Qe", instance->result_exhaustive.pval);
    printf("%8lld %4lld %.2f %.1f | %s %10.4f %6d | %s %10.4f | %s %10.4f | %s %10.4f\n",
               instance->N, instance->K, instance->k_power, instance->options_base.lambda,
               fourier_buf, instance->results_fourier[0].time, instance->results_fourier[0].L_stop,
               bagfft_buf, instance->results_bagfft[0].time,
               exact_multinom_buf, instance->result_exact_multinom.time,
               exhaustive_buf, instance->result_exhaustive.time
        );
}
static void csv_header(FILE* csv) {
    fprintf(csv, "N,K,k,lambda,algorithm,pvalue,time,Q,L,L_stop,B,converged,time_gamma,theta,eps_rel,err_rel_delta,err_rel_neff\n");
}
static void csv_line(FILE* csv, options_t* options, benchmark_instance_t* instance, benchmark_result_t* result, const char* method, int Q, int fourier, int bagFFT, int exact_multinom) {
    if (exact_multinom || !bagFFT) Q = -1;
    if (fourier) Q = (int)result->Q;

    char pval_buf[64];
    quadmath_snprintf(pval_buf, sizeof(pval_buf), "%.10Qe", result->pval);
    fprintf(csv,
        "%lld,%lld,%.2f,%.6f,%s,%s,%.6f,%d,%lld,%d,%lld,%d,%.6f,%.3e,%.3e,%.6e,%.6e\n",
        instance->N, instance->K, instance->k_power, options->lambda,
        method, pval_buf, result->time,
        Q, fourier? options->max_iter:-1, fourier? result->L_stop:-1, fourier? options->B:-1,
        result->converged, result->time_gamma, exact_multinom? instance->theta:NAN,
        fourier? options->eps_rel:NAN, fourier? result->err_rel_delta:NAN, fourier? result->err_rel_neff:NAN
    );
}
static void output_header(FILE* csv) {
    print_header();
    if (csv == NULL) return;
    csv_header(csv);
}
static void output_instance(FILE* csv, benchmark_instance_t* instance) {
    print_instance(instance);
    if (csv == NULL) return;
    if (RUN_FOURIER) {for (int i=0; i<instance->results_fourier_len; i++) csv_line(csv, &instance->options_fourier[i], instance, &instance->results_fourier[i], "Fourier", -1, 1, 0, 0);}
    if (RUN_BAGFFT) {for (int i=0; i<sizeof(Q_values)/sizeof(Q_values[0]); i++) csv_line(csv, &instance->options_base, instance, &instance->results_bagfft[i], "BagFFT", Q_values[i], 0, 1, 0);}
    if (RUN_EXACT_MULTINOM) {csv_line(csv, &instance->options_base, instance, &instance->result_exact_multinom, "ExactMultinom", -1, 0, 0, 1);}
    if (RUN_EXHAUSTIVE) {csv_line(csv, &instance->options_base, instance, &instance->result_exhaustive, "Exhaustive", -1, 0, 0, 0);}
}

static void main_benchmark() {
    options_t options;
    set_options_default(&options);
    options.verbose = 0;

    FILE* csv = fopen("output.csv", "w");
    if (csv == NULL) {
        perror("main_benchmark: failed to open output.csv");
        return;
    }
    const int_t N_len = sizeof(N_values)/sizeof(N_values[0]);
    const int_t K_len = sizeof(K_values)/sizeof(N_values[0]);
    const int_t k_power_len = sizeof(k_power_values)/sizeof(k_power_values[0]);
    const int_t lambda_len = sizeof(lambda_values)/sizeof(lambda_values[0]);
    const int_t instances_len = K_len*N_len*k_power_len*lambda_len;
    benchmark_instance_t* instances = malloc(instances_len*sizeof(benchmark_instance_t));

    output_header(NULL);
    int_t idx = 0;
    for (int_t i1=0; i1<N_len; i1++) {
    int run_exact_multinom = 1;
    for (int_t i2=0; i2<K_len; i2++) {
    for (int_t i3=0; i3<k_power_len; i3++) {
    for (int_t i4=0; i4<lambda_len; i4++) {
        instances[idx] = (benchmark_instance_t){.N = N_values[i1], .K = K_values[i2], .options_base = options, .k_power = k_power_values[i3]};
        instances[idx].options_base.lambda = lambda_values[i4];
        fill_log_support(&instances[idx]);
        instances[idx].idx = idx;

        init_benchmark_instance(&instances[idx]);
        run_benchmark_instance(&instances[idx], 1, 0, run_exact_multinom);
        if (instances[idx].result_exact_multinom.timed_out) run_exact_multinom = 0;
        output_instance(NULL, &instances[idx]);

        idx++;
    }
    }
    }
    }
    qsort(instances, instances_len, sizeof(benchmark_instance_t), support_cmp);

    printf("\n");
    output_header(csv);
    int run_exhaustive = 1;
    for (int_t i=0; i<instances_len; i++) {
        benchmark_instance_t* instance = &instances[i];
        if (instance->log_support/LOG(10.0) >= SUPPORT_CUTOFF) run_exhaustive = 0;
        run_benchmark_instance(instance, 0, run_exhaustive, 0);
        if (instance->result_exhaustive.timed_out) run_exhaustive = 0;
        output_instance(csv, instance);
    }
    for (int_t i=0; i<instances_len; i++) free_benchmark_instance(&instances[i]);

    fclose(csv);
    free(instances);
}
