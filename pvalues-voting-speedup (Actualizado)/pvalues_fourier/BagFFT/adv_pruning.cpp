#include "utils.h"
#include "adv_pruning.h"

//
// s = Score for which pruning is carried out
//
template<class score_t, class prob_t, class multi_pmf_t>
adv_pruning_t<score_t, prob_t, multi_pmf_t>::
adv_pruning_t(score_t* given_score, prob_t* given_prob, multi_pmf_t* given_multi_pmf, 
	      int given_offset, double s) {
  
  score = given_score; multi_pmf = given_multi_pmf; 
  offset = given_offset; prob = given_prob;
  log_pruned_pval = LOGZERO;  
  int N = score->get_N(); int K = score->get_K();
  kn2 = 1/((double) K*N*(N-1)/2.0); Qavg = 0;

  //
  // It_max is set to s+100 because we don't expect
  // entries greater than that to contribute significantly
  // to the p-value
  //
  It_min = NINT(s/score->get_step())-K/2-1-offset; 
  //It_max = NINT((s+100)/score->get_step())+K/2+1-offset;
  It_max = score->get_Q()-offset;
  It_prune_bound = NINT(s/score->get_step())+K/2+1-offset;
  
  min_firstK_I = new int*[K]; max_firstK_I = new int*[K];
  min_lastK_I = new int*[K+1]; max_lastK_I = new int*[K];
  
  min_firstK_I[0] = new int[N+1]; max_firstK_I[0] = new int[N+1];
  min_lastK_I[K-1] = new int[N+1]; max_lastK_I[K-1] = new int[N+1];
  
  //
  // Initialization
  //
  for(int n = 0;  n < N+1; n++) {
    
    min_firstK_I[0][n] = max_firstK_I[0][n] 
      = score->get_Ip(0, n) - offset; 
    min_lastK_I[K-1][n] = max_lastK_I[K-1][n] 
      = score->get_Ip(K-1, n);
  } 
  
  //
  // Recursively computing the bounds
  //
  for(int k = 1; k < K; k++) {
    
    min_firstK_I[k] = new int[N+1]; max_firstK_I[k] = new int[N+1];
    min_lastK_I[K-1-k] = new int[N+1]; max_lastK_I[K-1-k] = new int[N+1];

    //
    // For firstK we compute the entries for k from k-1 and
    // for lastK we compute the entries for K-1-k from K-k
    //
    for(int n = N; n >= 0; n--) {

      min_firstK_I[k][n] = min_firstK_I[k-1][n]+score->get_Ip(k, 0);
      max_firstK_I[k][n] = max_firstK_I[k-1][n]+score->get_Ip(k, 0);
      min_lastK_I[K-1-k][n] = min_lastK_I[K-k][n] + score->get_Ip(K-1-k, 0);
      max_lastK_I[K-1-k][n] = max_lastK_I[K-k][n]+score->get_Ip(K-1-k, 0);

      for(int i = 1; i <= n; i++) {

	min_firstK_I[k][n] = MIN(min_firstK_I[k][n], 
				 min_firstK_I[k-1][n-i]+score->get_Ip(k, i));
	max_firstK_I[k][n] = MAX(max_firstK_I[k][n], 
				 max_firstK_I[k-1][n-i]+score->get_Ip(k, i));
	
	min_lastK_I[K-1-k][n] = MIN(min_lastK_I[K-1-k][n], 
				    min_lastK_I[K-k][n-i]+score->get_Ip(K-1-k, i));
	max_lastK_I[K-1-k][n] = MAX(max_lastK_I[K-1-k][n], 
				    max_lastK_I[K-k][n-i]+score->get_Ip(K-1-k, i));
      }
    }
  }
}


