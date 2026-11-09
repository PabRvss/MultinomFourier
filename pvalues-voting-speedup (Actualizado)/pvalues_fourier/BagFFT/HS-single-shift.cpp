#include <math.h>
#include <iostream>
#include "utils.h"
#include "HS.h"

using namespace std;

#include "llr_score.h"
#include "multinom_prob.h"
#include "list.h"
#include "adv_pruning.h"
#include "multi_pmf.h"
#include "HS-options.h"
#include "HS-algorithm.h"

//
// Function: Computes and returns the p-value for the multinomial llr score s 
//           based on the shifted-hirji algorithm with lists and pruning.
//
// Parameters: N = Number of objects 
//             K = Number of bins 
//             s = LLR score
//             pu = Null multinomial distribution 
//             Q = Lattice size 
//             log_min_pval = log lower bound for p-value (set by the function)
//             log_max_pval = log upper bound for p-value (set by the function)
//
void HSsingle(int N, int K, double s, double *pu, int Q, double& log_min_pval, double& log_max_pval) {

  single_return_t result = 
    HSalgo<llr_score_t, multinom_prob_t<double>, list_t<double>, 
    adv_pruning_t<llr_score_t, multinom_prob_t<double>, multi_pmf_t<list_t<double> > >, 
    single_return_t>(N, K, s, pu, Q, set_shifted_p, get_single_return_value);

  compute_pvalues(result.log_pmf, result.step, Q, s, K, log_min_pval, log_max_pval);
  log_min_pval = log_sum(log_min_pval, result.log_pruned_pval);
  log_max_pval = log_sum(log_max_pval, result.log_pruned_pval);
}
