//#include "multinom_prob.h"
#include "poiss_prob.h"
#include "llr_score.h"
#include "convolution.h"
//#include "bagFFT-algorithm-multi-theta.h"
#include "bagFFT-algorithm.h"
#include "bagFFT.h"

//
// Function: Computes and returns the entire lattice pmf (log-values in an array of
//           size Q) for the multinomial llr score based on the bagFFT algorithm.
//
// Parameters: N = Number of objects 
//             K = Number of bins 
//             s = LLR score for which theta is optimized
//             pu = Null multinomial distribution 
//             Q = Lattice size 
//             step = Step size for the lattice (set by the function)
//             theta1 = Optimal theta1 shift (set by the function)
//             log_mgf = Log of the mgf of the pmf at theta1 (set by the function)
//             log_err_bnd = Bound on the absolute error for the pmf (set by the function)
// 
double* bagFFTarray(int N, int K, double s, double* pu, int Q, double& step, 
		    double& theta1, double& log_mgf, double*& log_err_bnd) {

//  return bagFFTalgo<llr_score_t, multinom_prob_t<double> >
  return bagFFTalgo_array<llr_score_t, poiss_prob_t<double> >
    (N, K, s, pu, Q, step, theta1, log_mgf, log_err_bnd, 
#if NR
     fftnr, conv_fftnr
#else
     fftw, conv_fftw
#endif
     );
}

//
// Function: Computes bounds for log of the p-value for the 
//           multinomial llr score based on the bagFFT algorithm.
//
// Parameters: N = Number of objects 
//             K = Number of bins 
//             s = LLR score for which theta is optimized
//             pu = Null multinomial distribution 
//             Q = Lattice size 
//             step = Step size for the lattice (set by the function)
//             theta1 = Optimal theta1 shift (set by the function)
//             log_mgf = Log of the mgf of the pmf at theta1 (set by the function)
//             log_err_bnd = Bound on the absolute error for the pmf (set by the function)
//             log_max_pval = Upper bound of log of the p-value (set by the function)
//             log_min_pval = Lower bound of log of the p-value (set by the function)
//
void bagFFTsingle(int N, int K, double s, double* pu, int Q, double& step, 
		  double& theta1, double& log_mgf, double*& log_err_bnd, 
		  double& log_max_pval, double& log_min_pval) {

  //return bagFFTalgo<llr_score_t, multinom_prob_t<double> >
  bagFFTalgo_single<llr_score_t, poiss_prob_t<double> >
    (N, K, s, pu, Q, step, theta1, log_mgf, log_err_bnd, 
#if NR
     fftnr, conv_fftnr,
#else
     fftw, conv_fftw,
#endif
     log_max_pval, log_min_pval);
}
