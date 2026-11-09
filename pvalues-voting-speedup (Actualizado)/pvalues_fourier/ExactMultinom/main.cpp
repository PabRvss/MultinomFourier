#include <cstdio>
#include <ctime>
#include <cmath>
#include <valarray>

typedef std::valarray<double> NumericVector;
typedef std::valarray<int> IntegerVector;

// Global variables used by the recursion
int len;

double * lprob_term;
double * llr_term;
double * chisq_term;

double * pr;
int * ipt;

bool * ipt0;
bool * bounds;

double stat_lprob;
double stat_prob;
double stat_llr;
double stat_chisq;

double newProbs_prob;
double newProbs_llr;
double newProbs_chisq;

// #include "multinom_pmf.cpp"
#include "multinom_llr.cpp"
// #include "multinom_all.cpp"

static inline double get_dt(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec)*1e-9;
}

int main() {
    int m = 6;
    int xi = 100;
    double theta = 1e-10;

    IntegerVector x(m);
    NumericVector p(m);
    for (int i=0; i<m; i++) {
        x[i] = xi;
        p[m-1-i] = std::pow(i+1, 1.0);
    }
    p /= p.sum();

    struct timespec start{}, end{};
    timespec_get(&start, TIME_UTC);
//    double res_pmf = multinom_test_pmf(x, p, theta);
    double res_llr = multinom_test_llr(x, p, theta);
//    result res_all = multinom_test_all(x, p, theta);
    timespec_get(&end, TIME_UTC);
    double dt = get_dt(start, end);

//    printf("pmf = %.6e\n", res_pmf);
    printf("llr = %.6e\n", res_llr);
//    printf("p: pmf=%.6e, llr=%.6e, chisq=%.6e\n", res_all.p_pmf, res_all.p_llr, res_all.p_chisq);
    printf("t: %.6f seconds\n", dt);

    return 0;
}
