#ifndef _ENTROPY_COMBINED_H_
#define _ENTROPY_COMBINED_H_

double* NE_NCpruned(int N, int K, int L, double s, double* pu, int Q, 
		    double& step);

double* HS_NCpruned(int N, int K, int L, double s, double* pu, int Q,
		    double& step);

double* NE_sFFT(int N, int K, int L, double s, double* pu, int Q, 
		double& step, double& theta, double& log_mgf);

double* HS_sFFT(int N, int K, int L, double s, double* pu, int Q, 
		double& step, double& theta, double& log_mgf);

double* HS_csFFT(int N, int K, int L, double s, double* pu, int Q, 
		 double& step, double& theta, double& log_mgf);

void stitched_LD_csFFT(int N, int K, int L, double s, double* pu, int Q, 
		       double& log_min_pval, double& log_max_pval);

double* bag_sFFT(int N, int K, int L, double s, double* pu, int Q, 
		 double& step, double& theta, double& log_mgf,
		 double*& log_err_bnd);

#endif
