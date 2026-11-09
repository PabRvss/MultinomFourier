#include "llr_score.h"
#include "multinom_prob.h"
#include "LD-algorithm.h"
#include "LD.h"

double basicLD(int N, int K, int L, double s, double* pu, int Q,
	       double& theta, double& log_mgf, int& reliable) {  

  return LDalgo<llr_score_t, multinom_prob_t<double> >(N, K, L, s, pu, Q, theta, log_mgf, reliable);
}
