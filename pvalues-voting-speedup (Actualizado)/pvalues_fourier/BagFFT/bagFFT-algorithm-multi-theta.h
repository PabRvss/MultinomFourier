#include <math.h>
#include <iostream>
#include <iomanip>

#include "utils.h"
#include "theta_fns.h"
#include "minimize.h"
#include "complex.h"

#include "extended_exponent.h"

using namespace std;

ee_t *op1;
ee_t *op2;
int size;
double target_mean;

double minFn(double theta) {

  ee_t sum;
  for(int i = 0; i < size; i++) {
    ee_t v(op1[i].log_get()-theta*i);
    sum += v;
  }

  return sum.log_get() + theta*target_mean;
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
double* bagFFTalgo(int N, int K, double s, double* pu, int Q, double& step, double& theta1, 
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
  double theta2[K+1]; target_mean = N*pu[0];
  op1 = new ee_t[2*N+1]; size = 2*N+1;
  op2 = new ee_t[N+1];
  double** log_p = prob.get_log_p_copy();

  for(int i = 0; i < N+1; i++)
    op1[i].log_set(log_p[0][i]);

  for(int k = 1; k < K; k++) {

    target_mean += N*pu[k];

    for(int i = 0; i < N+1; i++)
      op2[i].log_set(log_p[k][i]);

    conv(op2, op1, op1, N+1, 2*N+1);

    minimize(minFn, theta2[k]);
    
    for(int i = N+1; i < 2*N+1; i++)
      op1[i].log_set(LOGZERO);
  }

  theta2[0] = theta2[1];
  theta2[K] = theta2[K-1];

  theta2Fn<prob_t>::init(&prob, N2);
  double log_psi_err_bnd = theta2Fn<prob_t>::error_bound(theta2);

  //
  // Shifting and normalizing log_p based on theta2
  //
  double off_sum = 0; double exp_theta2[K][N+1];
  for(int k = 0; k < K; k++) {

    for(int i = 0; i < N+1; i++) {
      log_p[k][i] -= theta2[k]*i;
      exp_theta2[k][i] = exp((theta2[k]-theta2[k+1])*i);
    }

    double off = log_array_sum(log_p[k], N+1); off_sum += off;
    prob.norm2(k, off);
  }
  double log_theta2_offset = theta2[K]*N + off_sum + prob.log_const_factor();
  double log_phi_err_bnd = log_psi_err_bnd + prob.log_const_factor();

  //
  // Exponentiating log_p to get p
  //
  prob.set_p();
  double** p = prob.get_p_copy();

  if(DEBUG) {
    cout << "log(mgf): " << log_mgf << endl;
    cout << "theta1: " << theta1 << endl;
    cout << "theta2: ";
    for(int i = 0; i < K; i++)
      cout << theta2[i] << " ";
    cout << endl;
  }

  //
  // exp_i_omg_Ip[k][n] = exp(i*omg*Ip[k][n])
  // exp_i_l_omg_Ip[k][n] = exp(i*l*omg*Ip[k][n])
  // and is computed as exp_I_omg_Ip[k][n]^l during
  // the stage where the lth element of the characterisitc function
  // is being computed (see (***))
  //
  double omg = 2*pi/Q;  
  complex **exp_i_omg_Ip = new complex*[K];
  complex **exp_i_l_omg_Ip = new complex*[K];

  for (int k = 0; k < K; k++) {

    exp_i_omg_Ip[k] = new complex[N+1];
    exp_i_l_omg_Ip[k] = new complex[N+1];

    for (int x = 0; x < N+1; x++) {
      
      double angle = omg*score.get_Ip(k, x);
      exp_i_omg_Ip[k][x].set(cos(angle), sin(angle));
      exp_i_l_omg_Ip[k][x].set(1, 0);
    }
  }

  //
  // psi[l] = lth element of the characteristic function
  //
  complex* psi = new complex[Q];

  //
  // Computing the lth element of the characteristic function (***)
  //
  //for (int l = 0; l < 1+Q/2; l++) {
  for (int l = 0; l < Q; l++) {

    //
    // Initializing curr_psis
    //
    int k = 0; complex curr_psis[N2];
    for(int x = 0; x < N+1; x++) {
      curr_psis[x] = exp_i_l_omg_Ip[k][x]*p[k][x];
      exp_i_l_omg_Ip[k][x] *= exp_i_omg_Ip[k][x];
    } 

    for(k = 1; k < K; k++) {

      //
      // Initializing p_exp_k
      //
      complex p_exp_k[N2];                
      for(int x = 0; x < N+1; x++) { 
	p_exp_k[x] = exp_i_l_omg_Ip[k][x]*p[k][x];
	exp_i_l_omg_Ip[k][x] *= exp_i_omg_Ip[k][x];
      }

      complex next_psis[N2];  
      conv_fft(curr_psis, p_exp_k, next_psis, N2);

      //
      // Updating curr_psis
      //
      for(int n = 0; n < N2; n++)
	if(n < N+1)   
	  curr_psis[n] = next_psis[n]*exp_theta2[k][n];
	else
	  curr_psis[n].set(0, 0);         
    }
 
    //
    // Shifting curr_psis[N] to get psi[l]
    //
    psi[l] = curr_psis[N].shift(log_theta2_offset);

    //
    // Exploiting the symmetry for computing the DFT
    // of a real vector.
    //
    //if(l > 0)
    //  psi[Q-l] = psi[l].conj();
  }

  fft(psi, Q, -1);

  double max = 0;
  for(int It = 0; It < Q; It++)
    if(abs(psi[It].imag()) > max)
      max = abs(psi[It].imag());

  cout << max/Q << endl;
  cout << setprecision(15) << log(max) + log_mgf - log((double) Q) << endl;

  //
  // log_pmf = log lattice pmf (return value)
  //
  double* log_pmf = new double[Q];
  log_err_bnd = new double[Q];

  double log_ptheta_err_bnd = log_sum(log_phi_err_bnd, log(20*log((double) Q)*EPS)); 
  for(int It = 0; It < Q; It++) {
    log_pmf[It] = (psi[It].real() > 0 ? log(psi[It].real()) : LOGZERO) - log((double) Q) - 
      theta1*step*It + log_mgf;
    log_err_bnd[It] = log_ptheta_err_bnd - theta1*step*It + log_mgf;
  }

  return log_pmf;
}
