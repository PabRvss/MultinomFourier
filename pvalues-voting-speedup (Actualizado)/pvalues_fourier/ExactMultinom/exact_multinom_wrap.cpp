#include <cstdio>
#include <cmath>
#include <valarray>

#include "exact_multinom.h"

typedef std::valarray<double> NumericVector;
typedef std::valarray<int> IntegerVector;

// Global variables used by the recursion in multinom_llr.cpp (same names/shapes main.cpp declares).
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

#include "multinom_llr.cpp"
#include "multinom_llr_reduced.cpp"
#include "multinom_llr_modified.cpp"

extern "C" double exact_multinom_llr_pvalue(int K, const int* x0, const double* probs, double theta) {
    if (theta <= 0) theta = 0.0001;

    IntegerVector x(K);
    NumericVector p(K);
    for (int k = 0; k < K; k++) {
        x[k] = x0[k];
        p[k] = probs[k];
    }

    return multinom_test_llr(x, p, theta);
}

extern "C" double exact_multinom_llr_pvalue_reduced(int K, const int* x0, const double* probs, double theta) {
    if (theta <= 0) theta = 0.0001;

    IntegerVector x(K);
    NumericVector p(K);
    for (int k = 0; k < K; k++) {
        x[k] = x0[k];
        p[k] = probs[k];
    }

    return multinom_test_llr_reduced(x, p, theta);
}

extern "C" double exact_multinom_llr_pvalue_modified(int K, const int* x0, const double* probs, double theta, int* reliable) {
    if (theta <= 0) theta = 0.0001;

    IntegerVector x(K);
    NumericVector p(K);
    for (int k = 0; k < K; k++) {
        x[k] = x0[k];
        p[k] = probs[k];
    }

    bool cpp_reliable = true;
    double result = multinom_test_llr_modified(x, p, theta, &cpp_reliable);
    if (reliable) *reliable = cpp_reliable ? 1 : 0;
    return result;
}
