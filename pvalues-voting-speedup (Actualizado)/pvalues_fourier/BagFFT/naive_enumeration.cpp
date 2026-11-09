#include "utils.h"
#include "extended_exponent.h"

template<class score_t, class prob_t> 
double* NEalgo(int N, double *pu, int Q, double& step, 
	       void (*set_p)(prob_t& prob, score_t& score)) {

  int K = 4;
  score_t score(N, K, pu, Q); step = score.get_step();
  prob_t prob(N, K, pu);
  
  set_p(prob, score);

  ee_t* pmf = new ee_t[Q];
  for(int i = 0; i <= N; i++)
    for(int j = 0; j <= N-i; j++)
      for(int k = 0; k <= N-(i+j); k++) {
        int l = N-(i+j+k);
	double I = score.get_I(0, i) + score.get_I(1, j) +
	  score.get_I(2, k) + score.get_I(3, l);
	ee_t p = prob.get_p(0, i)*prob.get_p(1, j)*
	  prob.get_p(2, k)*prob.get_p(3, l);
	//pmf[NINT(I/step)] += p;
	pmf[int(I/step)] += p;
      }

  double* log_pmf = new double[Q];
  for(int It = 0; It < Q; It++)
    log_pmf[It] = pmf[It].log_get()+prob.log_const_factor();

  return log_pmf;
}

template<class score_t, class prob_t> 
double* NEalgo2(int N, int K, double *pu, int Q, double& step, 
		void (*set_p)(prob_t& prob, score_t& score)) {

  score_t score(N, K, pu, Q); step = score.get_step();
  prob_t prob(N, K, pu);
  
  set_p(prob, score);

  ee_t* pmf = new ee_t[Q];
  NErecurse(K-1, N, score, prob, 0.0, ee_t(0.0), pmf);
  
  double* log_pmf = new double[Q];
  for(int It = 0; It < Q; It++)
    log_pmf[It] = pmf[It].log_get()+prob.log_const_factor();

  return log_pmf;
}

template<class score_t, class prob_t> 
void NErecurse(int k, int N, score_t& score, prob_t& prob, double I, ee_t p, ee_t* pmf) {

  if(k == 0)
    pmf[NINT((I+score.get_I(k, N))/score.get_step())] += (p*prob.get_p(k, N));
  else
    for(int n = 0; n <= N; n++)
      NErecurse(k-1, N-n, score, prob, I+score.get_I(k, n), p*prob.get_p(k, n), pmf);
}

template<class score_t, class prob_t> 
double** multiNEalgo(int N, double *pu, int Q, double& step, 
		     void (*set_p)(prob_t& prob, score_t& score)) {

  int K = 4;
  score_t score(N, K, pu, Q); step = score.get_step();
  prob_t prob(N, K, pu);
  
  set_p(prob, score);

  double logn[N+1];
  logn[0] = 0;
  for(int i = 1; i < N+1; i++)
    logn[i] = log(i);

  ee_t** pmf = new ee_t*[N+1];
  for(int i = 0; i < N+1; i++)
    pmf[i] = new ee_t[Q];

  for(int i = 0; i <= N; i++)
    for(int j = 0; j <= N-i; j++)
      for(int k = 0; k <= N-(i+j); k++)
	for(int l = 0; l <= N-(i+j+k); l++) {
	  int n = i+j+k+l;
	  double I = score.get_I(0, i) + score.get_I(1, j) +
	    score.get_I(2, k) + score.get_I(3, l) + n*(logn[N] - logn[n]);
	  ee_t p = prob.get_p(0, i)*prob.get_p(1, j)*
	    prob.get_p(2, k)*prob.get_p(3, l);
	  //pmf[n][NINT(I/step)] += p;
	  pmf[n][int(I/step)] += p;
	}

  double** log_pmf = new double*[N+1];

  for(int i = 0; i < N+1; i++) {

    log_pmf[i] = new double[Q];
    for(int It = 0; It < Q; It++)
      log_pmf[i][It] = pmf[i][It].log_get() + prob.log_factorial(i);
  }

  return log_pmf;
}

#include "HS-options.h"
#include "llr_score.h"
#include "multinom_prob.h"

double* naive_enumeration(int N, double* pu, int Q, double& step) {

  return NEalgo<llr_score_t, multinom_prob_t<ee_t> >(N, pu, Q, step, set_ee_p);
  //return NEalgo2<llr_score_t, multinom_prob_t<ee_t> >(N, 4, pu, Q, step, set_ee_p);
}

double** multi_naive_enumeration(int N, double* pu, int Q, double& step) {

  return multiNEalgo<llr_score_t, multinom_prob_t<ee_t> >(N, pu, Q, step, set_ee_p);
}

