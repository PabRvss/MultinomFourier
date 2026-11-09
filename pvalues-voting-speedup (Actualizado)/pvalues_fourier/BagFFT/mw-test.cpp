#include <iostream>
#include <iomanip>
#include <ctime>
#include "utils.h"
#include "mw.h"

using namespace std;

int main(int argc, char** argv) {

  int argno = 1;
  int m = atoi(argv[argno++]);
  int n = atoi(argv[argno++]);
  int k = atoi(argv[argno++]);
  int c; int* size;

  if(argc > 4) {
    
    c = atoi(argv[argno++]);
    size = new int[c];
    for(int i = 0; i < c; i++)
      size[i] = atoi(argv[argno++]);
  }
  
  double* log_pmf; double log_pval; double log_min_pval;  
  double start = clock(); 

#ifdef MW_dp
  log_pmf = dp(m, n, k);
  compute_pvalues(log_pmf, 1.0, k+1, 0, 0, log_min_pval, log_pval); 
#endif

#ifdef MW_harding
  log_pmf = harding(m, n, k);
  compute_pvalues(log_pmf, 1.0, k+1, 0, 0, log_min_pval, log_pval); 
#endif

#ifdef MW_sFFT
  double* log_err_bnds;
  log_pmf = mw_sFFT(m, n, k, log_err_bnds); 
  compute_pvalues(log_pmf, 1.0, k+1, 0, 0, log_min_pval, log_pval); 
  
  double log_err_bnd; double log_min_err_bnd; 
  compute_pvalues(log_err_bnds, 1.0, k+1, 0, 0, log_min_err_bnd, log_err_bnd);
  cout << "err P = " << log_err_bnd << endl;
  cout << "Min guarantee (in decimal places): " << (log_pval-log_err_bnd)/log(10.0) << endl;
#endif

#ifdef MW_sFFTsingle
  double* log_err_bnds;
  mw_sFFTsingle(m, n, k, log_err_bnds, log_pval); 
  
  double log_err_bnd; double log_min_err_bnd; 
  compute_pvalues(log_err_bnds, 1.0, k+1, 0, 0, log_min_err_bnd, log_err_bnd);
  cout << "err P = " << log_err_bnd << endl;
  cout << "Min guarantee (in decimal places): " << (log_pval-log_err_bnd)/log(10.0) << endl;
#endif

#ifdef MW_csFFT
  //log_pmf = mw_refine_csFFT(m, n, k, atof(argv[4])); 
  double log_err_bnd; double log_min_err_bnd; 

  /*
  log_pmf = mw_csFFT(m, n, k, log_err_bnd, log_min_err_bnd); 
  compute_pvalues(log_pmf, 1.0, k+1, 0, 0, log_min_pval, log_pval);
  */

  log_pval = mw_refine_csFFT(m, n, k, atof(argv[4]), log_err_bnd, log_min_err_bnd); 
  //log_pval = mw_refine_csFFT(m, n, k, 0.0001, log_err_bnd, log_min_err_bnd); 

  cout << "err P = " << MAX(log_err_bnd, log_min_err_bnd) << endl;
  cout << "Min guarantee (in decimal places): " << (log_pval-MAX(log_err_bnd, log_min_err_bnd))/log(10.0) << endl;
#endif

#ifdef MW_csFFTsingle
  double log_err_bnd; double log_min_err_bnd; 
  mw_csFFTsingle(m, n, k, log_err_bnd, log_min_err_bnd, log_pval); 
  cout << "err P = " << MAX(log_err_bnd, log_min_err_bnd) << endl;
  cout << "Min guarantee (in decimal places): " << (log_pval-MAX(log_err_bnd, log_min_err_bnd))/log(10.0) << endl;
#endif

#ifdef MW_NA
  //int t = 3;
  //if(k < ((double) m)*n/2 - t*sqrt(((double) m)*n*(m+n+1)/12.0)) {

  //  cout << "P = " << endl;
  //  return 0;
  //}
  //cout << "Limit = " << ((double) m)*n/2 - t*sqrt(((double) m)*n*(m+n+1)/12.0) << endl;

  log_pval = mw_NA(m, n, k);

#endif

#ifdef MW_LD
  log_pval = mw_LD(m, n, k);
#endif

#ifdef MW_lattice
  log_pmf = mw_lattice(m, n, k, c, size);
  compute_pvalues(log_pmf, 1.0, k+1, 0, 0, log_min_pval, log_pval); 
#endif

#ifdef MW_tied_csFFT
  log_pmf = mw_tied_csFFT(m, n, k, c, size); 
  compute_pvalues(log_pmf, 1.0, k+1, 0, 0, log_min_pval, log_pval);
#endif

#ifdef MW_tied_sFFT
  log_pmf = mw_tied_sFFT(m, n, k, c, size); 
  compute_pvalues(log_pmf, 1.0, k+1, 0, 0, log_min_pval, log_pval);
#endif

#ifdef MW_tied_LD
  log_pval = mw_tied_LD(m, n, k, c, size);
#endif

  double finish = clock();
  cout << "Time = " << (finish*1.0/CLOCKS_PER_SEC - start*1.0/CLOCKS_PER_SEC) << endl;
  cout << "P = " << setprecision(20) << log_pval << " (" << exp(log_pval) << ")" << endl;
  cout << "EP format: " << setprecision(20) << exp(log_pval-int(log_pval)) << " "
       << int(log_pval)/log(10.0) << endl;

  return 0;
}
