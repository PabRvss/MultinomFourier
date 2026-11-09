#include <iostream>
#include <iomanip>
#include <time.h>
#include "utils.h"
#include "bagFFT.h"

using namespace std;

//
// Test program for testing bagFFT based functions
//
int main(int argc, char** argv) {

  //
  // Reading input parameters from the command line
  //
  int N; int K; double s; int Q; double* pu;
  read_llr_parameters(argc, argv, N, K, s, Q, pu);

  //
  // Running the algorithm and timing it
  //
  double start = clock();
  double step; double theta; double log_mgf; double* log_err_bnd;
  double log_max_pval; double log_min_pval;

#ifdef BAGFFT_single
  bagFFTsingle(N, K, s, pu, Q, step, theta, log_mgf, log_err_bnd, log_max_pval, log_min_pval);
#endif

#ifdef BAGFFT_array
  double* log_pmf = bagFFTarray(N, K, s, pu, Q, step, theta, log_mgf, log_err_bnd);
#endif

#ifdef CBAGFFT
  double* log_pmf = cbagFFTarray(N, K, s, pu, Q, step, theta, log_mgf, log_err_bnd);
#endif

  double finish = clock();
  cout << "Time = " << (finish*1.0/CLOCKS_PER_SEC - start*1.0/CLOCKS_PER_SEC) << endl;

  //
  // Computing the p-value and printing it
  //
#ifndef BAGFFT_single
  compute_theta_pvalues(log_pmf, step, Q, s, K, log_min_pval, log_max_pval, theta);
#endif
  cout << "P = " << setprecision(PRECISION) << log_max_pval << " (" << exp(log_max_pval) << ")" << endl;
  cout << "Min = " << setprecision(PRECISION) << log_min_pval << " (" << exp(log_min_pval) << ")" << endl; 

  //
  // Computing the error bounds and printing them
  //
  double log_max_pval_err_bnd; double log_min_pval_err_bnd;
  compute_theta_bounds(log_err_bnd, step, Q, s, K, log_min_pval_err_bnd, log_max_pval_err_bnd, theta);
  log_max_pval_err_bnd /= 2;   log_min_pval_err_bnd /= 2;

  cout << "err P = " << setprecision(PRECISION) << log_max_pval_err_bnd << " (" << exp(log_max_pval_err_bnd) << ")" << endl;
  cout << "err Min = " << setprecision(PRECISION) << log_min_pval_err_bnd << " (" << exp(log_min_pval_err_bnd) << ")" << endl; 

  cout << "Min guarantee (in decimal place): " << 
    MIN((log_max_pval-log_max_pval_err_bnd), (log_min_pval-log_min_pval_err_bnd))/log(10.0) << endl;  
}
