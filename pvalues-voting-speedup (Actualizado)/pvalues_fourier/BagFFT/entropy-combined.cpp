#include "utils.h"
#include "HS.h"
#include "bagFFT.h"
#include "entropy.h"
#include "LD.h"
#include "naive_enumeration.cpp"

double* NE_NCpruned(int N, int K, int L, double s, double* pu, int Q, 
		    double& step) {

  //
  // Computing the pmf for a single column
  //
  double* log_pmf = naive_enumeration(N, pu, Q, step);

  //
  // Shifted L-fold convolution
  //
  return NCpruned(log_pmf, Q, L, s);
}  

double* HS_NCpruned(int N, int K, int L, double s, double* pu, int Q, 
		    double& step) {

  //
  // Computing the pmf for a single column
  //
  double* log_pmf = HSarray(N, K, pu, Q, step);

  //
  // Shifted L-fold convolution
  //
  return NCpruned(log_pmf, Q, L, s);
}

double* NE_sFFT(int N, int K, int L, double s, double* pu, int Q, 
		double& step, double& theta, double& log_mgf) {

  //
  // Computing the pmf for a single column
  //
  double* log_pmf = naive_enumeration(N, pu, Q, step);

  //
  // Shifted L-fold convolution
  //
  theta = NOT_COMPUTED;
  return sFFT(log_pmf, L, s, Q, step, theta, log_mgf);
}

double* HS_sFFT(int N, int K, int L, double s, double* pu, int Q, 
		double& step, double& theta, double& log_mgf) {

  //
  // Computing the pmf for a single column
  //
  //double* log_pmf = HSarray(N, K, pu, Q, step);
  
  int* index; int size;
  double* log_pmf = HSlist(N, K, pu, Q, step, index, size);

  //
  // Shifted L-fold convolution
  //
  theta = NOT_COMPUTED;
  return sFFT(log_pmf, L, s, Q, step, theta, log_mgf);
}

double* HS_csFFT(int N, int K, int L, double s, double* pu, int Q, 
		 double& step, double& theta, double& log_mgf) {

  //
  // Computing the pmf for a single column
  //
  //double* log_pmf = HSarray(N, K, pu, Q, step);

  int* index; int size;
  double* log_pmf = HSlist(N, K, pu, Q, step, index, size);

  //
  // Shifted L-fold convolution
  //
  theta = NOT_COMPUTED;
  return csFFT(log_pmf, L, s, Q, step, theta, log_mgf);
}

void stitched_LD_csFFT(int N, int K, int L, double s, double* pu, int Q, 
			 double& log_min_pval, double& log_max_pval) {

  double theta; double log_mgf; int reliable;
  double log_pval = basicLD(N, K, L, s, pu, Q, theta, log_mgf, reliable);
  if(reliable)    
    log_min_pval = log_max_pval = log_pval;    
  else {

    double step; 
    double* log_pmfL = HS_csFFT(N, K, L, s, pu, Q, step, theta, log_mgf);  
    compute_theta_pvalues(log_pmfL, step, L*Q, s, L*K, log_min_pval, log_max_pval, theta);  
  }
}

double* bag_sFFT(int N, int K, int L, double s, double* pu, int Q, 
		 double& step, double& theta, double& log_mgf,
		 double*& log_err_bnd) {

  //
  // Computing the pmf for a single column
  //
  double* log_pmf = bagFFTarray(N, K, s/((double) L), pu, Q, step, 
				theta, log_mgf, log_err_bnd);

  double unshifted_err_bnd =
    log_sum(log((double) L) + log_err_bnd[0] - log_mgf, log(10*L*log((double) L*Q)*EPS));
  delete[](log_err_bnd);

  log_err_bnd = new double[L*Q];
  for(int i = 0; i < L*Q; i++)
    log_err_bnd[i] = -theta*step*i + L*log_mgf + unshifted_err_bnd; 

  //
  // Shifted L-fold convolution
  // 
  return sFFT(log_pmf, L, s, Q, step, theta, log_mgf);
}
