#include "base.cpp"

//
// Test program for testing functions for computing the p-value
// of the entropy score
//
int main(int argc, char** argv) {

  //
  // Reading input parameters from the command line
  //
  int N; int K; int L; double s; int Q; double* pu;
  read_entropy_parameters(argc, argv, N, K, L, s, Q, pu);

  //
  // Running the algorithm and timing it
  //
  double start = clock();   

  double step; double theta; double log_mgf; double* log_pmfL; 
  double log_max_pval; double log_min_pval; double* log_err_bnd;

#ifdef LD
  int reliable;
  double log_pval = basicLD(N, K, L, s, pu, Q, theta, log_mgf, reliable);
#endif

#ifdef STITCHED
  stitched_LD_csFFT(N, K, L, s, pu, Q, log_min_pval, log_max_pval);
#endif

#ifdef NE_NCPRUNED
  char pmf_filename[100];
  sprintf(pmf_filename, "pmf2/%d-%d-%d-%d-%0.4f", N, K, L, Q, pu[0]);
  FILE* pmf_file = fopen(pmf_filename, "r");
  if(!pmf_file) {

    log_pmfL = NE_NCpruned(N, K, L, s, pu, Q, step);
    pmf_file = fopen(pmf_filename, "w");
    fprintf(pmf_file, "%lf\n", step);
    for(int i = 0; i < L*Q; i++) 
      fprintf(pmf_file, "%0.20lg\n", log_pmfL[i]);
    fclose(pmf_file);
  }
  else {
    
    log_pmfL = new double[L*Q]; 
    fscanf(pmf_file, "%lf", &step);
    step = N*log(1/pu[0])/(Q-1);
    for(int i = 0; i < L*Q; i++)
      fscanf(pmf_file, "%lg", &log_pmfL[i]); 
    
    fclose(pmf_file);
  }
#endif

#ifdef HS_NCPRUNED
  char pmf_filename[100];
  sprintf(pmf_filename, "pmf/%d-%d-%d-%d-%0.4f", N, K, L, Q, pu[0]);
  FILE* pmf_file = fopen(pmf_filename, "r");
  if(!pmf_file) {

    log_pmfL = HS_NCpruned(N, K, L, s, pu, Q, step);  
    pmf_file = fopen(pmf_filename, "w");
    fprintf(pmf_file, "%lf\n", step);
    for(int i = 0; i < L*Q; i++) 
      fprintf(pmf_file, "%0.20lg\n", log_pmfL[i]);
    fclose(pmf_file);
  }
  else {

    log_pmfL = new double[L*Q]; 
    fscanf(pmf_file, "%lf", &step);
    step = N*log(1/pu[0])/(Q-1);
    for(int i = 0; i < L*Q; i++)
      fscanf(pmf_file, "%lg", &log_pmfL[i]); 

    fclose(pmf_file);
  }
#endif

#ifdef NE_SFFT
  log_pmfL = NE_sFFT(N, K, L, s, pu, Q, step, theta, log_mgf);  
#endif

#ifdef HS_SFFT
  log_pmfL = HS_sFFT(N, K, L, s, pu, Q, step, theta, log_mgf);  

  log_err_bnd = new double[L*Q];
  for(int i = 0; i < L*Q; i++)
    log_err_bnd[i] = log(5*L*(N*K*log((double) N)/log(2.0) + 20*log((double) L*Q))/log(2.0))+log(EPS)-theta*step*i+L*log_mgf;
#endif

#ifdef HS_CSFFT
  log_pmfL = HS_csFFT(N, K, L, s, pu, Q, step, theta, log_mgf);  
#endif

#ifdef BAG_SFFT
  log_pmfL = bag_sFFT(N, K, L, s, pu, Q, step, theta, log_mgf, log_err_bnd);  
#endif

  double finish = clock();
  cout << "Time = " << setprecision(5) << ((finish-start)/(1.0*CLOCKS_PER_SEC)) << " " << (finish - start) << endl;

  //
  // Computing the p-value and printing it
  //  

#ifdef LD
  cout << "P = " << setprecision(20) << log_pval << " (" << exp(log_pval) << ")" << endl;
  return 0;
#endif

#ifdef NE_NCPRUNED
  compute_pvalues(log_pmfL, step, L*Q, s, L, log_min_pval, log_max_pval);
#endif

#ifdef HS_NCPRUNED
  compute_pvalues(log_pmfL, step, L*Q, s, L*K, log_min_pval, log_max_pval);
#endif

#ifndef HS_NCPRUNED
#ifndef NE_NCPRUNED
#ifndef STITCHED
  compute_theta_pvalues(log_pmfL, step, L*Q, s, L*K, log_min_pval, log_max_pval, theta);
#endif  
#endif  
#endif
  
  cout << "P = " << setprecision(20) << log_max_pval << " (" << exp(log_max_pval) << ")" << endl;
  cout << "Min = " << setprecision(20) << log_min_pval << " (" << exp(log_min_pval) << ")" << endl; 

#ifndef BAG_SFFT
#ifndef HS_SFFT
  return 0;
#endif
#endif

  double log_max_pval_err_bnd; double log_min_pval_err_bnd;
  compute_theta_err_bnds(log_err_bnd, step, L*Q, s, L*K, log_min_pval_err_bnd, log_max_pval_err_bnd, theta);

  cout << "err P = " << setprecision(20) << log_max_pval_err_bnd << " (" << exp(log_max_pval_err_bnd) << ")" << endl;
  cout << "err Min = " << setprecision(20) << log_min_pval_err_bnd << " (" << exp(log_min_pval_err_bnd) << ")" << endl;   
  cout << "Min guarantee (in decimal places): " << 
    (log_max_pval < -0.01 ? MIN((log_max_pval-log_max_pval_err_bnd), (log_min_pval-log_min_pval_err_bnd))/log(10.0) : 16) << endl;  

  return 0;
}
