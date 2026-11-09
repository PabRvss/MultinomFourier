#ifndef _ADV_PRUNING_H_
#define _ADV_PRUNING_H_

#include "utils.h"

//
// Class for doing aggressive pruning of a multi_pmf_t (see multi_pmf.h)
// in Hirji's algorithm. The "pruning" based on this class
// is meant to generate a single p-value. 
// For more basic pruning for computing the entire pmf
// the class basic_pruning_t in basic_pruning.h should be used.
//
// Note: this class should probably be derived from basic_pruning_t
//
// score_t = The type of the score for which the pruning is being
//           computed 
// prob_t = The type of the probability distribution for which the pruning is being
//           computed 
// multi_pmf_t = The instantiated type of multi_pmf_t which
//               is being pruned
//
template<class score_t, class prob_t, class multi_pmf_t>
class adv_pruning_t {

 private:
  
  //
  // Score, probability and multi_pmf for which the pruning
  // is being carried out
  // 
  score_t* score; prob_t* prob; multi_pmf_t* multi_pmf; 

  //
  // min_firstK_I[k][n] = Minimum possible partial lattice score
  //                      with n objects in the first k bins
  // max_firstK_I[k][n] = Maximum possible partial lattice score
  //                      with n objects in the first k bins
  // min_lastK_I[k][n] = Minimum possible partial lattice score
  //                     with n objects in the last k bins
  // max_lastK_I[k][n] = Maximum possible partial lattice score
  //                     with n objects in the last k bins
  //
  int** min_firstK_I; int** max_firstK_I;
  int** min_lastK_I; int** max_lastK_I;

  double Qavg; double kn2;

  //
  // log_pruned_pval = Log of the sum of pruned pmf entries
  //                 = Log of the probability that should be added to
  //                   the p-value computed with the remaining pmf entries
  // offset = Offset for the score
  //
  double log_pruned_pval; int offset; 

  //
  // It_min = Minimum index for a pmf entry that contributes to the p-value
  // It_min = Maximum index for a pmf entry that contributes significantly to the p-value
  // It_prune_bound = Bound such that if all extensions of a partial score are 
  //                  greater than it then the corresponding pmf entry can be pruned
  //
  int It_min; int It_max; int It_prune_bound;

 public:
  
  //
  // s = Score for which pruning is carried out
  //
  adv_pruning_t(score_t* given_score, prob_t* given_prob, multi_pmf_t* given_multi_pmf, 
		int given_offset, double s); 
		
  //
  // Functions for iterating over the pmf.  
  // The appropriate pmf functions with the computed bounds are
  // called.
  //
  inline int start(int k, int n, int i) 
    { return multi_pmf->pmf[n-i].start(min_firstK_I[k][n-i]); }
  inline int not_last(int k, int n, int i)  
    { return multi_pmf->pmf[n-i].not_last(max_firstK_I[k][n-i]); }
  inline int next(int k, int n, int i) { 

    Qavg += kn2;
    int N = score->get_N();
    int It = multi_pmf->pmf[n-i].next(max_firstK_I[k][n-i]); 
    if(k == score->get_K()-1) return It;

    while(not_last(k, n, i)) {

      //
      // If this condition is satisfied then the
      // pmf entry at It is s.t. it will always contribute to the
      // p-value => it can be pruned
      //
      if(It >= It_prune_bound - min_lastK_I[k+1][N-n+i]) {

	//
	// The p-value contribution is computed by unshifting
	// the pmf entry at It by both the shifts and multiplying
	// it by the probability contribution for the
	// remaining bins
	//
	log_pruned_pval = 
	  log_sum(log_pruned_pval, log(multi_pmf->pmf[n-i][It]) - prob->get_theta1()*(It+offset)*score->get_step() 
		  + prob->get_theta2()*(n-i) + prob->log_const_rem_factor(k+1, n-i)); 
	
	multi_pmf->pmf[n-i].remove_cur();
      }
      else 
	//
	// If this condition is not satisfied then the entry at It
	// will not contribute significantly to the p-value
	// => we can skip it
	//
	if(It  >= It_min - max_lastK_I[k+1][N-n+i] 
	   && It <= It_max - min_lastK_I[k+1][N-n+i]) 
	  break;      

      It = multi_pmf->pmf[n-i].next(max_firstK_I[k][n-i]);       
    }
    
    return It;
  }

  inline double get_Qavg() { return Qavg; }
  inline double get_log_pruned_pval() { return log_pruned_pval; }

  ~adv_pruning_t() {

    for(int k = 0; k < score->get_K(); k++) {

      delete[](min_firstK_I[k]); delete[](max_firstK_I[k]);
      delete[](min_lastK_I[k]); delete[](max_lastK_I[k]);
    }

    delete[](min_lastK_I[score->get_K()]); delete[](max_lastK_I[score->get_K()]);
    delete[](min_lastK_I); delete[](max_lastK_I);
    delete[](min_firstK_I); delete[](max_firstK_I);
  }
};

#include "adv_pruning.cpp"
  
#endif
