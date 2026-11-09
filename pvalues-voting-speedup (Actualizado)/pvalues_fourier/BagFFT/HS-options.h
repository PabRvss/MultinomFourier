#ifndef _HS_OPTIONS_H_
#define _HS_OPTIONS_H_

#include "utils.h"
#include "llr_score.h"
#include "multinom_prob.h"
#include "extended_exponent.h"
#include "basic_pruning.h"
#include "adv_pruning.h"
#include "multi_pmf.h"

//
// Hirji's DP algorithm (see HS-algorithm.h) can
// be executed in either the shifted or the extended-exponent
// mode with set_shifted_p and set_ee_p being the corresponding
// "set_p" functions. These functions are currently implemented
// for the case where we have multinomial probabilities and
// the llr score.
//

inline void set_shifted_p(multinom_prob_t<double>& prob, llr_score_t& score) {

  prob.shift1(1, score.get_Ip_copy(), score.get_step());
  prob.shift2(-(log((double) prob.get_N())-1));
  prob.set_p();
}

inline void set_ee_p(multinom_prob_t<ee_t>& prob, llr_score_t& score) {

  prob.set_p();
}

//
// Hirji's DP algorithm (see HS-algorithm.h) can be
// configured to return an array of values
// in the pmf or a single p-value using the corresponding
// functions get_array_return_value and get_single_return_value.
//
 
struct array_return_t {

  double step;
  double* log_pmf;
  int* index;
  int size;
};

//
// The options for p_t are double and ee_t.
// The options for pmf_t are array<p_t> and list<p_t>
//
template<class pmf_t, class p_t>
array_return_t get_array_return_value(multi_pmf_t<pmf_t>& multi_pmf, 
			  int N, double step, int offset, multinom_prob_t<p_t>& prob, 
			  basic_pruning_t<llr_score_t, multinom_prob_t<p_t>, multi_pmf_t<pmf_t> >& pruning) {

  array_return_t return_value;
  return_value.step = step;
  int Q = multi_pmf.get_max_range()+offset;
  int K = prob.get_K();
  return_value.log_pmf = new double[Q];
  for(int i = 0; i < Q; i++)
    return_value.log_pmf[i] = LOGZERO;

  return_value.index = new int[Q]; return_value.size = 0;

  //
  // Iterating through multi_pmf.pmf[N] to recover the result pmf
  //
  for(int It = pruning.start(K-1, N, 0); pruning.not_last(K-1, N, 0); It = pruning.next(K-1, N, 0)) 
    if(It >= -offset) {
      return_value.log_pmf[It + offset] = log(multi_pmf.pmf[N][It]) + prob.log_const_factor() 
	- prob.get_theta1()*(It+offset)*step + prob.get_theta2()*N;
      return_value.index[return_value.size++] = It + offset;
    }

  return return_value;
}

struct single_return_t {

  double step;
  double* log_pmf;
  double log_pruned_pval;
};

//
// The options for p_t are double and ee_t.
// The options for pmf_t are array<p_t> and list<p_t>
//
template<class pmf_t, class p_t>
single_return_t get_single_return_value(multi_pmf_t<pmf_t>& multi_pmf, 
			  int N, double step, int offset, multinom_prob_t<p_t>& prob, 
			  adv_pruning_t<llr_score_t, multinom_prob_t<p_t>, multi_pmf_t<pmf_t> >& pruning) {

  single_return_t return_value;
  return_value.step = step; return_value.log_pruned_pval = pruning.get_log_pruned_pval();
  cout << "Pruned contribution: " << return_value.log_pruned_pval << endl;

  int Q = multi_pmf.get_max_range()+offset;
  int K = prob.get_K();
  return_value.log_pmf = new double[Q];
  for(int i = 0; i < Q; i++)
    return_value.log_pmf[i] = LOGZERO;

  //
  // Iterating through multi_pmf.pmf[N] to recover the result pmf
  //
  for(int It = pruning.start(K-1, N, 0); pruning.not_last(K-1, N, 0); It = pruning.next(K-1, N, 0))
    if(It >= -offset) 
      return_value.log_pmf[It + offset] = log(multi_pmf.pmf[N][It]) + prob.log_const_factor() 
	- prob.get_theta1()*(It+offset)*step + prob.get_theta2()*N;

  return return_value;
}

//
// New option - returning a multiarray
//

struct multiarray_return_t {

  double step;
  double** log_pmf;
};

//
// The options for p_t are double and ee_t.
// The options for pmf_t are array<p_t> and list<p_t>
//
template<class pmf_t, class p_t>
multiarray_return_t get_multiarray_return_value(multi_pmf_t<pmf_t>& multi_pmf, 
					   int N, double step, int offset, multinom_prob_t<p_t>& prob, 
					   basic_pruning_t<llr_score_t, multinom_prob_t<p_t>, multi_pmf_t<pmf_t> >& pruning) {

  multiarray_return_t return_value;
  return_value.step = step;
  int Q = multi_pmf.get_max_range()+offset;
  int K = prob.get_K();
  return_value.log_pmf = new double*[N+1];
  for(int i = 0; i < N+1; i++) {
    
    return_value.log_pmf[i] = new double[Q];
    for(int j = 0; j < Q; j++)
      return_value.log_pmf[i][j] = LOGZERO;
  }

  for(int i = 0; i < N+1; i++)
    for(int It = pruning.start(K-1, i, 0); pruning.not_last(K-1, i, 0); It = pruning.next(K-1, i, 0))
      if(It + offset + NINT(i/step*(log(N)-log(i))) >= 0)
	return_value.log_pmf[i][It + offset + NINT(i/step*(log(N)-log(i)))] = log(multi_pmf.pmf[i][It]) + prob.log_factorial(i) 
	  - prob.get_theta1()*(It+offset)*step + prob.get_theta2()*i;

  return return_value;
}

#endif
