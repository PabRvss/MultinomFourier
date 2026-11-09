#include <math.h>
#include <iostream>
#include <iomanip>

#include "utils.h"
#include "theta_fns.h"
#include "minimize.h"
#include "complex.h"

using namespace std;

//
// Function: Computes and returns the DFT of the lattice pmf (complex_t array of size Q)
//           based on the bagFFT algorithm.
//
// Parameters: N = Number of objects 
//             K = Number of bins 
//             s = LLR score for which theta is optimized
//             pu = Null multinomial distribution 
//             Q = Lattice size 
//             step = Step size for the lattice (set by the function)
//             theta1 = Optimal theta1 shift (set by the function)
//             log_mgf = Log of the mgf of the pmf at theta1 (set by the function)
//             log_phi_err_bnd = Error bound for phi (set by the function)
//             fft = Function for computing the FFT
//             conv_fft = Function for computing the convolution using FFTs
//
template<class score_t, class prob_t>
complex_t* bagFFTalgo_internal(int N, int K, double s, double* pu, int Q, double& step, double& theta1, 
			   double& log_mgf, double& log_phi_err_bnd, 
			   void (*fft)(complex_t* a, int n, int sign),
			   void (*conv_fft)(complex_t* a , complex_t* b, complex_t* result, int n)) {

#if NR
  Q = RND_UP_TO_POWER_2(Q);
#endif

  score_t score(N, K, pu, Q);
  prob_t prob(N, K, pu);
  step = score.get_step();
  
  //
  // Minimizing theta1
  //
  theta1Fn<score_t, prob_t>::init(&score, &prob, s);
  log_mgf = minimize(theta1Fn<score_t, prob_t>::minimizationFn, theta1); 
  log_mgf += s*theta1;
  //theta1 -= 0.01;
  //log_mgf = theta1Fn<score_t, prob_t>::minimizationFn(theta1)+s*theta1;
 
  //
  // Shifting and normalizing log_p based on theta1
  //
  prob.shift1(theta1, score.get_Ip_copy(), score.get_step());
  prob.norm1(log_mgf/((double) K));

  //
  // N2 = 2*N+1
  //
  int N2 = RND_UP_TO_POWER_2(2*N+1);

  //
  // Minimizing theta2
  //
  double theta2; theta2Fn<prob_t>::init(&prob, N2, score.get_Ip_copy(), score.get_step(), pu);
  double log_psi_err_bnd = minimize(theta2Fn<prob_t>::minimizationFn, theta2);
  //theta2 -= 0.01;

  //
  // Shifting and normalizing log_p based on theta2
  //
  prob.shift2(theta2);
  double off_sum = 0; double** log_p = prob.get_log_p_copy();
  for(int k = 0; k < K; k++) {

    double off = log_array_sum(log_p[k], N+1); off_sum += off;
    prob.norm2(k, off);
  }
  double log_theta2_offset = theta2*N + off_sum + prob.log_const_factor();
  log_phi_err_bnd = log_psi_err_bnd + prob.log_const_factor();

  //
  // Exponentiating log_p to get p
  //
  prob.set_p();
  double** p = prob.get_p_copy();

  if(DEBUG) {
    cout << setprecision(PRECISION) << "log(mgf): " << log_mgf << endl;
    cout << "theta1: " << theta1 << endl;
    cout << "theta2: " << theta2 << endl;
  }

  double omg = 2*pi/Q;
  
  //
  // exp_i_omg[x] = exp(i*omg*x)
  //
  complex_t *exp_i_omg = new complex_t[Q];
  for(int x = 0; x < Q; x++)
    exp_i_omg[x].set(cos(omg*x), sin(omg*x));

  //
  // psi[l] = lth element of the characteristic function
  //
  complex_t *psi = new complex_t[Q];
  //
  // Computing the lth element of the characteristic function
  //
  for(int l = 0; l < 1+Q/2; l++) {

    //
    // Initializing curr_psis
    //
    int k = 0; complex_t *curr_psis = new complex_t[N2];
    for(int x = 0; x < N+1; x++)
      curr_psis[x] = 
	//exp_i_omg[((unsigned long) l*(score.get_Ip(k, x)+Q))%Q]*p[k][x];
	exp_i_omg[mod(score.get_Ip(k, x)+Q, l, Q)]*p[k][x];

    for(k = 1; k < K; k++) {

      //
      // Initializing p_exp_k
      //
      complex_t *p_exp_k = new complex_t[N2];                
      for(int x = 0; x < N+1; x++) 
	p_exp_k[x] = 
	  //exp_i_omg[((unsigned long) l*(score.get_Ip(k, x)+Q))%Q]*p[k][x]; 
	  exp_i_omg[mod(score.get_Ip(k, x)+Q, l, Q)]*p[k][x];

      complex_t *next_psis = new complex_t[N2];  
      conv_fft(curr_psis, p_exp_k, next_psis, N2);

      //
      // Updating curr_psis
      //
      for(int n = 0; n < N2; n++)
	if(n < N+1)   
	  curr_psis[n] = next_psis[n];
	else
	  curr_psis[n].set(0, 0);         

      // clean up
      delete [] p_exp_k;
      delete [] next_psis;
    }
 
    //
    // Shifting curr_psis[N] to get psi[l]
    //
    psi[l] = curr_psis[N].shift(log_theta2_offset);

    //
    // Exploiting the symmetry for computing the DFT
    // of a real vector.
    //
    if(l > 0)
      psi[Q-l] = psi[l].conj();

    //clean up
    delete [] curr_psis;
  }

  return psi;
}

//
// Function: Computes and returns the entire lattice pmf (log-values
//           in an array of size Q) based on the bagFFT algorithm.
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
//             fft = Function for computing the FFT
//             conv_fft = Function for computing the convolution using FFTs
//
template<class score_t, class prob_t>
double* bagFFTalgo_array(int N, int K, double s, double* pu, int Q, double& step, double& theta1, 
		    double& log_mgf, double*& log_err_bnd, 
		    void (*fft)(complex_t* a, int n, int sign),
		    void (*conv_fft)(complex_t* a , complex_t* b, complex_t* result, int n)) {

  double log_phi_err_bnd;
  complex_t* psi = bagFFTalgo_internal<score_t, prob_t>(N, K, s, pu, Q, step, theta1, log_mgf, log_phi_err_bnd, fft, conv_fft);
  //for(int It = 0; It < Q; It++) 
  //  cout << setprecision(PRECISION) << psi[It].real() << " ";
  //cout << endl;

  fft(psi, Q, -1);

  //
  // log_pmf = log lattice pmf (return value)
  //
  double* log_pmf = new double[Q];
  log_err_bnd = new double[Q];

  double log_ptheta_err_bnd = log_sum(log_phi_err_bnd, log(10*log((double) Q)/log(2.0)*EPS)); 
  for(int It = 0; It < Q; It++) {
    log_pmf[It] = (psi[It].real() > 0 ? log(psi[It].real()) : LOGZERO) - log((double) Q) - 
      theta1*step*It + log_mgf;
    //cout << setprecision(PRECISION) << log(fabs(psi[It].real())) - log((double) Q) + log_mgf << " ";
    log_err_bnd[It] = log_ptheta_err_bnd - theta1*step*It + log_mgf;
  } //cout << endl;
  
  return log_pmf;
}

//
// Function: Computes the bounds for log of the p-value based on the bagFFT algorithm.
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
//             fft = Function for computing the FFT
//             conv_fft = Function for computing the convolution using FFTs
//             log_max_pval = Upper bound for log of the p-value (set by the function)
//             log_min_pval = Lower bound for log of the p-value (set by the function)
//
template<class score_t, class prob_t>
void bagFFTalgo_single(int N, int K, double s, double* pu, int Q, double& step, double& theta1,
			 double& log_mgf, double*& log_err_bnd, 
			 void (*fft)(complex_t* a, int n, int sign),
			 void (*conv_fft)(complex_t* a , complex_t* b, complex_t* result, int n), 
			 double& log_max_pval, double& log_min_pval) {
  
  double log_phi_err_bnd;
  complex_t* psi = bagFFTalgo_internal<score_t, prob_t>(N, K, s, pu, Q, step, theta1, log_mgf, log_phi_err_bnd, fft, conv_fft);

  log_err_bnd = new double[Q];
  double log_ptheta_err_bnd = log_sum(log_phi_err_bnd, log(10*log((double) Q)/log(2.0)*EPS)); 
  for(int It = 0; It < Q; It++)
    log_err_bnd[It] = log_ptheta_err_bnd - theta1*step*It + log_mgf;

  compute_phi_pvalue(psi, step, Q, s/step, (theta1 > 0 ? Q : 0), K, log_min_pval, log_max_pval, theta1, log_mgf);

  if(theta1 <= 0) {

    log_min_pval = log(1-exp(log_min_pval));
    log_max_pval = log(1-exp(log_max_pval));
  }
}

