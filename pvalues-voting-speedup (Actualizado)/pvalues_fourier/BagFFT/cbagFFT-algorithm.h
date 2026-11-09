#include <math.h>
#include <iostream>
#include <iomanip>

#include "utils.h"
#include "theta_fns.h"
#include "minimize.h"
#include "complex.h"

using namespace std;

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
double* cbagFFTalgo(int N, int K, double s, double* pu, int Q, double& step, double& theta1, 
		    double& log_mgf, double*& log_err_bnd, 
		    void (*fft)(complex* a, int n, int sign),
		    void (*conv_fft)(complex* a , complex* b, complex* result, int n)) {

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

  //
  // Shifting and normalizing log_p based on theta1
  //
  prob.shift1(theta1, score.get_Ip_copy(), score.get_step());
  prob.norm1(log_mgf/((double) K));

  //
  // N2 = 2*N+1
  //
#if NR
  int N2 = RND_UP_TO_POWER_2(2*N+1);
#else
  int N2 = 2*N+1;
#endif

  //
  // Minimizing theta2
  //
  double theta2; theta2Fn<prob_t>::init(&prob, N2, score.get_Ip_copy(), score.get_step(), pu);
  double log_psi_err_bnd = minimize(theta2Fn<prob_t>::minimizationFn, theta2);

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
  double log_phi_err_bnd = log_psi_err_bnd + prob.log_const_factor();

  //
  // Exponentiating log_p to get p
  //
  prob.set_p();
  double** p = prob.get_p_copy();

  if(DEBUG) {
    cout << "log(mgf): " << log_mgf << endl;
    cout << "theta1: " << theta1 << endl;
    cout << "theta2: " << theta2 << endl;
  }

  double omg = 2*pi/Q;
  
  //
  // exp_i_omg[x] = exp(i*omg*x)
  //
  complex *exp_i_omg = new complex[Q];
  for(int x = 0; x < Q; x++)
    exp_i_omg[x].set(cos(omg*x), sin(omg*x));

  //
  // psi[l] = lth element of the characteristic function
  //
  complex *psi = new complex[Q];
  complex *copy;

  int originalQ = Q; double prev_log_pval = LOGZERO;

  for(Q = 32; Q <= originalQ; Q *= 2) {

    int ratio = originalQ/Q;

    for(int l = Q/2-1; l >= 0; l--) { 
      psi[2*l] = psi[l]; 
      if(l%2 != 0) psi[l].set(0, 0); 
    }

    //
    // Computing the lth element of the characteristic function
    //
    for(int l = 0; l < 1+Q/2; l++) {
    
      if(psi[l].real() != 0 || psi[l].imag() != 0)
	continue;

      //
      // Initializing curr_psis
      //
      int k = 0; complex *curr_psis = new complex[N2];
      for(int x = 0; x < N+1; x++)
	curr_psis[x] = exp_i_omg[mod(score.get_Ip(k, x)+originalQ, l, Q)*ratio]*p[k][x];
      
      for(k = 1; k < K; k++) {
	
	//
	// Initializing p_exp_k
	//
	complex *p_exp_k = new complex[N2];                
	for(int x = 0; x < N+1; x++) 
	  p_exp_k[x] = exp_i_omg[mod(score.get_Ip(k, x)+originalQ, l, Q)*ratio]*p[k][x];
	
      complex *next_psis = new complex[N2];  
      conv_fft(curr_psis, p_exp_k, next_psis, N2);
      
      //
      // Updating curr_psis
      //
      for(int n = 0; n < N2; n++)
	if(n < N+1)   
	  curr_psis[n] = next_psis[n];
	else
	  curr_psis[n].set(0, 0);         


      // Clean up
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

      // Clean up
      delete [] curr_psis;
    }

    copy = new complex[Q];
    for(int l = 0; l < Q; l++)
      copy[l] = psi[l];

    fft(copy, Q, -1);

    double log_pval = LOGZERO;
    for(int It = (int) floor(s/step); It < originalQ; It++)
      log_pval = log_sum(log_pval, (copy[It%Q].real() > 0 ? log(copy[It%Q].real()) : LOGZERO) - log((double) Q) - 
			 theta1*step*It + log_mgf);

    if(fabs(prev_log_pval-log_pval) < 0.1)
      break;

    delete[](copy);    
    prev_log_pval = log_pval;
  }

  Q = (Q > originalQ ? originalQ : Q);
  fft(psi, Q, -1);
  cout << "Qprime: " << Q << endl;

  //
  // log_pmf = log lattice pmf (return value)
  //
  double* log_pmf = new double[originalQ];
  log_err_bnd = new double[originalQ];

  double log_ptheta_err_bnd = log_sum(log_phi_err_bnd, log(10*log((double) Q)/log(2.0)*EPS)); 
  for(int It = 0; It < originalQ; It++)
  {  //if(It > s/step - Q/2 && It < s/step + Q/2) {
      log_pmf[It] = (psi[It%Q].real() > 0 ? log(psi[It%Q].real()) - log((double) Q) - 
		     theta1*step*It + log_mgf : LOGZERO);
      log_err_bnd[It] = log_ptheta_err_bnd - theta1*step*It + log_mgf;
    }
//else {
//     log_pmf[It] = LOGZERO;
//    log_err_bnd[It] = LOGZERO;
//  }

  return log_pmf;
}
