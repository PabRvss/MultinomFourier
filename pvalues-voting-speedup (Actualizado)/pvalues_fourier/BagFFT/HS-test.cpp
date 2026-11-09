#include <iostream>
#include <iomanip>
#include <time.h>
#include "utils.h"
#include "HS.h"
#include "naive_enumeration.cpp"

using namespace std;

//
// Test program for testing Hirji/HS's algorithm based functions
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
  double step; double log_max_pval; double log_min_pval; 
  double* log_pmf;

  //
  // Array version
  //
#ifdef ARRAY
  log_pmf = HSarray(N, K, pu, Q, step);
#endif

  //
  // List version
  //
#ifdef LIST
  int* index; int size;
  log_pmf = HSlist(N, K, pu, Q, step, index, size);
#endif

  //
  // Single p-value version
  //
#ifdef SINGLE
  HSsingle(N, K, s, pu, Q, log_min_pval, log_max_pval);
#endif

  //
  // Multiarray version
  //
#ifdef MULTIARRAY
  //double** result = HSmultiarray(N, K, pu, Q, step);
  //double** result = multi_naive_enumeration(N, pu, Q, step);
  log_pmf = naive_enumeration(N, pu, Q, step);
#endif

  double finish = clock();
  cout << "Time = " << (finish*1.0/CLOCKS_PER_SEC - start*1.0/CLOCKS_PER_SEC) << endl;

#ifdef MULTIARRAY
  /*  
  for(int i = 0; i < N+1; i++) {
    compute_pvalues(result[i], step, Q, s, K, log_min_pval, log_max_pval);
    cout << "For N=" << i << ": " << setprecision(PRECISION) << log_max_pval << " " << log_min_pval << endl;
    
    double sum = LOGZERO;
    for(int j = 0; j < Q; j++) { 
      cout << result[i][j] << " ";
      sum = log_sum(sum, result[i][j]);
    }
    cout << ": " << sum << endl;
    
  }
  
  return 0;
*/
#endif

  //
  // Computing the p-value and printing it
  //
#ifndef SINGLE
  compute_pvalues(log_pmf, step, Q, s, K, log_min_pval, log_max_pval);
#endif
  cout << "P = " << setprecision(PRECISION) << log_max_pval << " (" << exp(log_max_pval) << ")" << endl;
  cout << "Min = " << setprecision(PRECISION) << log_min_pval << " (" << exp(log_min_pval) << ")" << endl; 
}
