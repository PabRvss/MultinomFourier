#ifndef _LD_ALGORITHM_H_
#define _LD_ALGORITHM_H_

#include <math.h>
#include <iostream>
#include "utils.h"
#include "theta_fns.h"
#include "minimize.h"

//
// Note: Need to create a .h file for this
//
#include "gamma-prob.c"

using namespace std;

template<class score_t, class prob_t>
double LDalgo(int N, int K, int L, double s, double* pu, int Q, 
	      double& theta, double& log_mgf, int& reliable) { 
  
  score_t score(N, K, pu, Q);
  prob_t prob(N, K, pu);
  double step = score.get_step();

  //
  // Minimizing theta
  //
  theta1Fn<score_t, prob_t>::init(&score, &prob, s/((double) L));
  log_mgf = minimize(theta1Fn<score_t, prob_t>::minimizationFn, theta); 
  log_mgf += s*theta/((double) L);

  //
  // Computing sigma
  //
  double sigma = theta1Fn<score_t, prob_t>::get_sigma(theta, -score.get_min_t_ent());
  reliable = (MIN(L*score.get_max_ent()-s, s) > 3*sqrt((double) L)*sigma);
  
  if(DEBUG)
    cout << "theta: " << theta << endl
	 << "sigma: " << sigma << endl;

  //
  // Computing the result from the LD formula
  //
  double lambda = fabs(theta)*sigma*sqrt((double) L);
  double integrated_result = integrate_normal(lambda, 0, 1);
  double log_pval = pow(lambda, 2)/2.0 + integrated_result + L*log_mgf-theta*s;
  return (theta > 0 ? log_pval : log(1-exp(log_pval)));
}
 
#endif
