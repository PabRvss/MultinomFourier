#ifndef _MW_H_
#define _MW_H_

double* dp(int m, int n, int k);
double* harding(int m, int n, int k);

double* mw_sFFT(int m, int n, int k, double*& log_err_bnd);
void mw_sFFTsingle(int m, int n, int k, double*& log_err_bnd, double& log_pval);

double* mw_csFFT(unsigned long int m, unsigned long int n, int k,
		 double& log_upper_err_bound, double& log_lower_err_bound);
void mw_csFFTsingle(unsigned long int m, unsigned long int n, int k, 
		    double& log_upper_err_bound, double& log_lower_err_bound, double& log_pval);

double mw_refine_csFFT(int m, int n, int k, double a,
		       double& log_lower_err_bound, double& log_upper_err_bound);

double* mw_bound_csFFT(int m, int n, int k);
double mw_NA(int m, int n, int k);
double mw_LD(int m, int n, double k);

double* mw_lattice(int m, int n, int k, int c, int* size);
double* mw_tied_sFFT(int m, int n, int k, int c, int* size);
double* mw_tied_csFFT(int m, int n, int k, int c, int* size);
double mw_tied_LD(int m, int n, double k, int c, int* size);

#endif
