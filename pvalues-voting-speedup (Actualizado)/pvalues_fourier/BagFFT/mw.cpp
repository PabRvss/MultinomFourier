#include <iostream>
#include <strstream>
#include <fstream>
#include <string>
#include <iomanip>
#include "math.h"
#include "utils.h"
#include "theta_fns.h"
#include "minimize.h"
#include "convolution.h"
#include "complex.h"
#include "gamma-prob.c"
#include "mw.h"

using namespace std;

//
// Ensures that m, n and k are in the correct
// range before returning the contents of the
// "pmf" f
//
inline double fread(double** f, int m, int n, int k) {

  return ((k == 0 && m >= 0 && n >= 0) ? 0 : ((m <= 0 || n <= 0 || k < 0 || k > m*n) ? LOGZERO : f[m-1][k]));
}

//
// Compute the pmf for MW by dynamic
// programming
//
double* dp(int m, int n, int k) {

  int Q = m*n+1;
  int N = m+n;

  double** current = new double*[m];
  double** next = new double*[m];
  for(int i = 0; i < m; i++) {
    current[i] = new double[Q];
    next[i] = new double[Q];
    for(int j = 0; j < Q; j++)
      current[i][j] = next[i][j] = LOGZERO;
  }

  double** log_fac = new double*[N];
  for(int i = 0; i < N; i++) {
    log_fac[i] = new double[i];
    log_fac[i][0] = log(i+1);
    for(int j = 1; j < i; j++)
      log_fac[i][j] = log_fac[i][j-1] + log(i+1-(j+1)+1)-log(j+1);
  }

  for(int i = 2; i <= N; i++) {
    for(int j = 1; j <= MIN(m, i-1); j++) {  
      next[j-1][0] = 0; 
      for(int k = 1; k < j*(i-j)+1; k++)
	next[j-1][k] = log_sum(fread(current, j-1,  i-j, k), fread(current, j, i-j-1, k-j));
  
      if(j <= i-j) {

	ostrstream filename; filename << "pmf/" << j << "-" << (i-j);
	ofstream file(filename.str()); file << setprecision(20) << next[j-1][0] - log_fac[i-1][j-1] << " "; 
	for(int k = 1; k < j*(i-j)+1; k++)
	  file << next[j-1][k] - log_fac[i-1][j-1] << " "; 

	file << endl;
	file.close();
      }
    }

    double** temp = current; current = next; next = temp;
  }

  double* result = new double[Q];
  for(int i = 0; i < Q; i++)
    result[i] = current[m-1][i]-log_fac[N-1][m-1];

  return result;
}

#include "log_value.h"
typedef double harding_t;

//
// Harding's algorithm for computing
// the pmf for the MW statistic
//
double* harding(int m, int n, int k) {

  harding_t* f = new harding_t[k+1];
  f[0] = 1;

  if(n < k) {

    for(int t = n+1; t <= MIN(m+n, k); t++) 
      for(int u = k; u >= t; u--)
	f[u] -= f[u-t];
  }

  for(int s = 1; s <= MIN(m, k); s++) 
    for(int u = s; u <= k; u++)
      f[u] += f[u-s];

  double log_CNm = 0;
  for(int j = 1; j <= m; j++)
    log_CNm += log(n+j)-log(j);

  double* result = new double[k+1];
  for(int u = 0; u <= k; u++) 
    result[u] = log(f[u])-log_CNm;

  return result;
}


#define CARE 1

//
// Computes the characteristic function for a
// given theta and Q'
//
complex* compute_phi_theta_Q(int m, int n, double theta, int Qp) {

  double* exp_theta_k = new double[m+n+1];
#if CARE
  double* sinh_theta_k = new double[m+n+1];
  double* cosh_theta_k = new double[m+n+1];
#endif

  for(int j = 0; j < m+n+1; j++) {
    exp_theta_k[j] = exp(theta*j);
#if CARE
    sinh_theta_k[j] = sinh(theta*j/2.0);
    cosh_theta_k[j] = cosh(theta*j/2.0);
#endif
  }

  double omg = 2*pi/Qp;
  complex *exp_i_omg_x = new complex[Qp];

#if CARE
  double* sin_omg_x = new double[Qp];
  double* cos_omg_x = new double[Qp];
#endif

  for(int x = 0; x < Qp; x++) {
#if CARE
    sin_omg_x[x] = sin(omg*x/2.0);
    cos_omg_x[x] = cos(omg*x/2.0);
#endif
    exp_i_omg_x[x].set(cos(omg*x), sin(omg*x));
  }

#if CARE
  int limit1 = MIN(m+1, int(1/fabs(theta)-n+1));
  int limit2 = MIN(m+1, int(1/fabs(theta)+1));
#endif

  complex* Dcpmf = new complex[Qp];
  for(int l = 0; l < 1+Qp/2; l++) {

    Dcpmf[l].set(1, 0);

    int j=1;

#if CARE
    for(; j < limit1; j++) {
      int lnj = mod(l, n+j, Qp);
      int lj = mod(l, j, Qp);
      Dcpmf[l] *= (complex(cos_omg_x[lnj], sin_omg_x[lnj])*complex(cos_omg_x[lnj]*sinh_theta_k[n+j], sin_omg_x[lnj]*cosh_theta_k[n+j]))/
	(complex(cos_omg_x[lj], sin_omg_x[lj])*complex(cos_omg_x[lj]*sinh_theta_k[j], sin_omg_x[lj]*cosh_theta_k[j]))*
	(sinh_theta_k[j]/sinh_theta_k[n+j]);
    }
    
    for(; j < limit2; j++) {
      int lj = mod(l, j, Qp);
      Dcpmf[l] *= ((exp_i_omg_x[mod(l, n+j, Qp)]*exp_theta_k[n+j]-1)/(exp_theta_k[n+j]-1))/
	(complex(cos_omg_x[lj], sin_omg_x[lj])*complex(cos_omg_x[lj]*sinh_theta_k[j], sin_omg_x[lj]*cosh_theta_k[j])/sinh_theta_k[j]);
    }
#endif

    for(; j <= m; j++)
      Dcpmf[l] *= ((exp_i_omg_x[mod(l, n+j, Qp)]*exp_theta_k[n+j]-1)/
		   (exp_i_omg_x[mod(l, j, Qp)]*exp_theta_k[j]-1)*
		   ((exp_theta_k[j]-1)/(exp_theta_k[n+j]-1)));
 
    if(l > 0)
      Dcpmf[Qp-l] = Dcpmf[l].conj();
  }

  return Dcpmf;
}

//
// sFFT in the untied case
//
double* mw_sFFT(int m, int n, int k, double*& log_err_bnd) {

  double theta; mw_thetaFn::init(m, n, k);  
  double log_mgf = minimize(mw_thetaFn::minimizationFn, theta);
  log_mgf += theta*k;

  int Q = m*n + 1; 

#if NR
  int Qp = RND_UP_TO_POWER_2(Q);
#else
  int Qp = (Q < 200000 ? Q : RND_UP_TO_POWER_2(Q));
#endif

  if(DEBUG) {
    cout << "theta: " << theta << endl; 
    cout << "log(MGF): " << log_mgf << endl;
    cout << "Qp: " << Qp << endl;
  }

  complex* Dcpmf = compute_phi_theta_Q(m, n, theta, Qp);

#if NR
  fftnr(Dcpmf, Qp, -1);
#else
  if(Qp < 200000)
    fftw(Dcpmf, Qp, -1);
  else
    fftnr(Dcpmf, Qp, -1);
#endif 

  double* log_pmf = new double[Q];
  log_err_bnd = new double[Q];

  double error = (100*m + 20*log(Q))*EPS;
  for(int j = 0; j < Q; j++) {
    log_pmf[j] = (Dcpmf[j].real() <= 0 ? LOGZERO : log(Dcpmf[j].real()) - theta*j + log_mgf -log((double) Qp));
    log_err_bnd[j] = log(error) - theta*j + log_mgf;
  }

  return log_pmf;
}

//
// Version of sFFT to compute a single p-value
//
void mw_sFFTsingle(int m, int n, int k, double*& log_err_bnd, double& log_pval) {

  double theta; mw_thetaFn::init(m, n, k);  
  double log_mgf = minimize(mw_thetaFn::minimizationFn, theta);
  log_mgf += theta*k;

  int Q = m*n + 1; int Qp = Q;

  if(DEBUG) {
    cout << "theta: " << theta << endl; 
    cout << "log(MGF): " << log_mgf << endl;
    cout << "Qp: " << Qp << endl;
  }

  complex* Dcpmf = compute_phi_theta_Q(m, n, theta, Qp);

  log_err_bnd = new double[Q];
  double error = 160*m*EPS;
  for(int j = 0; j < Q; j++) 
    log_err_bnd[j] = log(error) - theta*j + log_mgf -log((double) Qp);
  
  //for(int j = 0; j < Q; j++) 
  //  cout << Dcpmf[j].real() << "," << Dcpmf[j].imag() << " ";
  //cout << endl;

  double log_min_pval;
  compute_phi_pvalue(Dcpmf, 1, Qp, k+1, 0, 0, log_min_pval, log_pval, theta, log_mgf);

  /*
  complex phi_sum;
  double omg = 2*pi/Qp;

  for(int It = 0; It < Qp; It++) {
    complex x(-theta, -omg*It);
    complex x_Qj(-theta*(k+1), -omg*(k+1)*It);
    phi_sum += Dcpmf[It]*exp_minus_one(x_Qj)/exp_minus_one(x);  
  }

  log_pval = log(abs(phi_sum.real())) + log_mgf - log(Qp);
  */
}

//
// Computes the error bounds for mw-csCF in the untied case
//
void compute_csCF_bounds(int m, int n, int k, int Qp, double theta,
			 double& log_lower_err_bound, double& log_upper_err_bound) {

  log_lower_err_bound = LOGZERO;
  if(k-Qp/2 > 0) {

    double alpha; mw_thetaFn::init(m, n, k-Qp/2);  
    log_lower_err_bound = minimize(mw_thetaFn::minimizationFn, alpha);
  }

  double log_upper_err_bound1 = LOGZERO;
  if(k-Qp > 0) {

    double gamma; mw_thetaFn::init(m, n, k-Qp);  
    log_upper_err_bound1 = minimize(mw_thetaFn::minimizationFn, gamma) - theta*Qp;
  }

  double log_upper_err_bound2 = LOGZERO;
  double log_upper_err_bound3 = LOGZERO;

  if(k+Qp/2 <= m*n) {

    double delta; mw_thetaFn::init(m, n, k+Qp/2);  
    log_upper_err_bound2 = minimize(mw_thetaFn::minimizationFn, delta) + theta*Qp;
    log_upper_err_bound3 = log_upper_err_bound2 + log((exp(-delta)-exp(-delta*(Qp/2+1)))/(1-exp(-delta))) + log(1-exp(theta));
  }  

  log_upper_err_bound = log_sum(log_upper_err_bound1, log_sum(log_upper_err_bound2, log_upper_err_bound3));
}

//
// csFFT in the untied case
//
double* mw_csFFT(unsigned long int m, unsigned long int n, int k, 
		 double& log_upper_err_bound, double& log_lower_err_bound) {

  double theta; mw_thetaFn::init(m, n, k);  
  double log_mgf = minimize(mw_thetaFn::minimizationFn, theta);
  log_mgf += theta*k;

  double sigma = exp(mw_thetaFn::get_log_sigma(theta));
  unsigned long int Q = m*n + 1; 

#if NR
  long int Qp = RND_UP_TO_POWER_2((long int) ceil(MIN(Q, 6*sigma)));
#else
  int Qp = (int) ceil(MIN(Q, 6*sigma));
#endif

  if(DEBUG) {
    cout << "theta: " << theta << endl; 
    cout << "log(MGF): " << log_mgf << endl;
    cout << "sigma: " << sigma << endl;
    cout << "Qp: " << Qp << endl;
  }

  complex* Dcpmf = compute_phi_theta_Q(m, n, theta, Qp);

#if NR
  fftnr(Dcpmf, Qp, -1);
#else
  if(Qp < 200000)
    fftw(Dcpmf, Qp, -1);
  else
    fftnr(Dcpmf, Qp, -1);
#endif

  double* log_pmf = new double[Q];
  for(long int j = 0; j < Q; j++) log_pmf[j] = LOGZERO;

  for(long int j = MAX(0, (int) k-Qp/2); j <= k; j++)
    if(Dcpmf[j%Qp].real() > 0)
      log_pmf[j] = log(Dcpmf[j%Qp].real()) - theta*j + log_mgf -log((double) Qp);

  compute_csCF_bounds(m, n, k, Qp, theta, log_lower_err_bound, log_upper_err_bound);

  return log_pmf;
}

//
// csCF in the untied case
//
void mw_csFFTsingle(unsigned long int m, unsigned long int n, int k, 
		    double& log_upper_err_bound, double& log_lower_err_bound, double& log_pval) {

  double theta; mw_thetaFn::init(m, n, k);  
  double log_mgf = minimize(mw_thetaFn::minimizationFn, theta);
  log_mgf += theta*k;

  double sigma = exp(mw_thetaFn::get_log_sigma(theta));
  unsigned long int Q = m*n + 1; 

  int Qp = (int) ceil(MIN(Q, 6*sigma));

  if(DEBUG) {
    cout << "theta: " << theta << endl; 
    cout << "log(MGF): " << log_mgf << endl;
    cout << "sigma: " << sigma << endl;
    cout << "Qp: " << Qp << endl;
  }

  complex* Dcpmf = compute_phi_theta_Q(m, n, theta, Qp);

  double log_min_pval;
  compute_phi_pvalue(Dcpmf, 1, Qp, MAX(0, int(k-Qp/2)), k+1, 0, log_min_pval, log_pval, theta, log_mgf);
  compute_csCF_bounds(m, n, k, Qp, theta, log_lower_err_bound, log_upper_err_bound);
}

double min_theta;
double min_log_mgf;

double f(double k) {

  double theta;
  k = (int) k;
  mw_thetaFn::k = (int) k; 
  double log_mgf = minimize(mw_thetaFn::minimizationFn, theta); 
  log_mgf += theta*k;

  return fabs(log_mgf-log(0.01)-(theta-min_theta)*k-min_log_mgf);
}

//
// Version of csFFT that minimizes error bounds
//
double* mw_bound_csFFT(int m, int n, int k) {

  double theta; mw_thetaFn::init(m, n, k);  
  double log_mgf = minimize(mw_thetaFn::minimizationFn, theta);
  log_mgf += theta*k; 

  double exp_theta_k[2*m+n+1];
  for(int j = 0; j < 2*m+n+1; j++)
    exp_theta_k[j] = exp(theta*j);

  min_theta = theta;
  min_log_mgf = log_mgf;
  double bound_k;
  minimize(f, bound_k);
  cout << "Bound k: " << bound_k << endl;

  int Q = m*n + 1; 
  double diff = fabs(bound_k-k);

#if NR
  int Qp = RND_UP_TO_POWER_2((int) (MIN(Q, k+diff)-MAX(0, k-diff)));
#else
  int Qp = (int) (MIN(Q, k+diff)-MAX(0, k-diff));
#endif

  if(DEBUG) {
    cout << "theta: " << theta << endl; 
    cout << "log(MGF): " << log_mgf << endl;
    cout << "Qp: " << Qp << endl;
  }

  double omg = 2*pi/Qp;
  complex *exp_i_omg_x = new complex[Qp];
  for(int x = 0; x < Qp; x++)
    exp_i_omg_x[x].set(cos(omg*x), sin(omg*x));

  complex *Dcpmf = new complex[Qp];
  for(int l = 0; l < 1+Qp/2; l++) {

    Dcpmf[l].set(1, 0);
    for(int j = 1; j <= m; j++)
      Dcpmf[l] *= ((exp_i_omg_x[(l*(n+j))%Qp]*exp_theta_k[n+j]-1)/
		   (exp_i_omg_x[(l*j)%Qp]*exp_theta_k[j]-1)*
		   ((exp_theta_k[j]-1)/(exp_theta_k[n+j]-1)));

    if(l > 0)
      Dcpmf[Qp-l] = Dcpmf[l].conj();
  }

#if NR
  fftnr(Dcpmf, Qp, -1);
#else
  fftw(Dcpmf, Qp, -1);
#endif

  double* log_pmf = new double[Q];
  for(int j = 0; j < Q; j++) log_pmf[j] = LOGZERO;

  for(int j = MAX(0, (int) k-Qp/2); j <= k; j++)
    if(Dcpmf[j%Qp].real() > 0)
      log_pmf[j] = log(Dcpmf[j%Qp].real()) - theta*j + log_mgf -log((double) Qp);

  return log_pmf;
}

//
// Version of csFFT that refines the lattice
//
double mw_refine_csFFT(int m, int n, int k, double a,
		       double& log_lower_err_bound, double& log_upper_err_bound) {

  double theta; mw_thetaFn::init(m, n, k);  
  double log_mgf = minimize(mw_thetaFn::minimizationFn, theta);
  log_mgf += theta*k;

  double exp_theta_k[m+n+1];
  for(int j = 0; j < m+n+1; j++)
    exp_theta_k[j] = exp(theta*j);

  int Q = m*n + 1; 
  int Qp = RND_UP_TO_POWER_2(Q);

  if(DEBUG) {
    cout << "theta: " << theta << endl; 
    cout << "log(MGF): " << log_mgf << endl;
    cout << "Qp: " << Qp << endl;
  }

  complex *exp_i_omg_x = new complex[Qp];
  double omg = 2*pi/Qp;
  for(int x = 0; x < Qp; x++)
    exp_i_omg_x[x].set(cos(omg*x), sin(omg*x));

  complex *Dcpmf = new complex[Qp];
  complex *cpmf = new complex[Qp];
  int Qmax = Qp; double log_pval = LOGZERO;

  double sigma = exp(mw_thetaFn::get_log_sigma(theta));
  int Qstart = MAX(8, RND_UP_TO_POWER_2((int) ceil(MIN(Q, 2*sigma))));

  for(Qp = Qstart; Qp <= Qmax; Qp *= 2) {

    int ratio = Qmax/Qp;

    /*
    double omg = 2*pi/Qp;
    if(Qp == Qstart)
      for(int x = 0; x < Qp; x++)
	exp_i_omg_x[x*ratio].set(cos(omg*x), sin(omg*x));
    else
      for(int x = 0; x < Qp/2; x++)
	exp_i_omg_x[(2*x+1)*ratio].set(cos(omg*(2*x+1)), sin(omg*(2*x+1)));
    */

    int index = 0;
    for(int l = (Qp > Qstart); l < Qp/2.0 + (Qp == Qstart); l += 1+(Qp > Qstart)) {

      Dcpmf[index].set(1, 0);
      for(int j = 1; j <= m; j++)
	Dcpmf[index] *= ((exp_i_omg_x[mod(l, n+j, Qp)*ratio]*exp_theta_k[n+j]-1)/
			 (exp_i_omg_x[mod(l, j, Qp)*ratio]*exp_theta_k[j]-1)*
			 ((exp_theta_k[j]-1)/(exp_theta_k[n+j]-1)));

      if(l > 0)
	Dcpmf[(Qp > Qstart ? Qp/2-1 : Qp)-index] = Dcpmf[index].conj();
      index++;
    }

    double log_min_pval;
    double old_log_pval;

    if(Qp == Qstart) {
      /*
#if NR
      fftnr(Dcpmf, Qp, -1);
#else
      if(Qp < 200000)
	fftw(Dcpmf, Qp, -1);
      else
	fftnr(Dcpmf, Qp, -1);
#endif

      for(int j = 0; j < Qp; j++)
	cpmf[j] = Dcpmf[j];

      for(int j = MAX(0, (int) k-Qp/2); j <= k; j++)
	if(cpmf[j%Qp].real() > 0)
	  log_pval = log_sum(log_pval, log(cpmf[j%Qp].real()) - theta*j + log_mgf - log((double) Qp));
      */
      
      for(int j = 0; j < Qp; j++)
	cpmf[j] = Dcpmf[j];

      compute_phi_pvalue(cpmf, 1, Qp, MAX(0, int(k-Qp/2)), k+1, 0, log_min_pval, log_pval, theta, log_mgf);      
      old_log_pval = log_pval;
    }
    else {

      /*
#if NR
      fftnr(Dcpmf, Qp/2, -1);
#else
      if(Qp/2 < 200000)
	fftw(Dcpmf, Qp/2, -1);
      else
	fftnr(Dcpmf, Qp/2, -1);
#endif

      for(int j = 0; j < Qp/2; j++) {
	cpmf[Qp/2+j] = cpmf[j] - Dcpmf[j]*complex(cos((-2*pi*j)/Qp), sin((-2*pi*j)/Qp));
	cpmf[j] = cpmf[j] + Dcpmf[j]*complex(cos((-2*pi*j)/Qp), sin((-2*pi*j)/Qp));
      }

      log_pval = LOGZERO;
      for(int j = MAX(0, (int) k-Qp/2); j <= k; j++)
	if(cpmf[j%Qp].real() > 0)
	  log_pval = log_sum(log_pval, log(cpmf[j%Qp].real()) - theta*j + log_mgf - log((double) Qp));

      */

      for(int j = Qp/2-1; j >= 0; j--) {
	cpmf[2*j] = cpmf[j];
	cpmf[2*j+1] = Dcpmf[j];
      }
      
      //for(int j = 0; j < Qp; j++) 
      //cout << cpmf[j].real() << "," << cpmf[j].imag() << " ";
      //cout << endl;

      compute_phi_pvalue(cpmf, 1, Qp, MAX(0, int(k-Qp/2)), k+1, 0, log_min_pval, log_pval, theta, log_mgf);
      
      //cout << Qp << " " << old_log_pval << " " << log_pval << endl;
      if(fabs(log_pval-old_log_pval) < a)
	break;

      old_log_pval = log_pval;
    }
  }

  /*
  double* log_pmf = new double[Q];
  for(int j = 0; j < Q; j++) log_pmf[j] = LOGZERO;
  */

  if(Qp > Qmax) Qp = Qmax; 
  cout << "Final Q: " << Qp << endl;

  /*
  for(int j = MAX(0, (int) k-Qp/2); j <= k; j++)
    if(cpmf[j%Qp].real() > 0)
      log_pmf[j] = log(cpmf[j%Qp].real()) - theta*j + log_mgf - log((double) Qp);
  
  return log_pmf;
  */

  compute_csCF_bounds(m, n, k, Qp, theta, log_lower_err_bound, log_upper_err_bound);
  return log_pval;
}

//
// Normal Approximation
//
double mw_NA(int m, int n, int k) {

  double dm = (double) m; double dn = (double) n;
  double x = (k+0.5-(dm*dn/2))/sqrt(dm*dn*(dm+dn+1)/12.0);

  //double p = dm/(dm+dn);
  //double c20 = -6*(1-pow(p, 5)-pow(1-p, 5))/(25*pow(p*(1-p), 2));
  //double H3x = pow(x, 3)-3*x;

  double integrate = integrate_normal(-x, 0, 1);
  double Phix = exp(-x*x/2-integrate)/sqrt(2*pi);
  double var = dm*dn*(dm+dn+1);
  double c1 = -(dm*dm + dn*dn + dm*dn + dm + dn)*(3*x-pow(x, 3))/(20*var);
  double c2 = -(2*(pow(dm, 4) + pow(dn, 4)) + 4*dm*dn*(dm*dm + dn*dn) + 
		6*pow(dm*dn, 2) + 4*(pow(dm, 3) + pow(dn, 3)) + 
		7*dm*dn*(dm + dn) + (dm*dm + dn*dn) + 2*dm*dn - (dm+dn))*
    (15*x - 10*pow(x, 3) + pow(x, 5))/(210*var*var);
  double c3 = pow(dm*dm + dn*dn + dm*dn + dm + dn, 2)*
    (105*x-105*pow(x, 3)+21*pow(x, 5)-pow(x, 7))/(800*var*var);

  //return integrate + log(1+Phix*(c1+c2+c3));
  return integrate;
}

//
// Large deviation variants
//
double mw_LD(int m, int n, double k) {

  k += 0.5;
  k = ((double) m)*n-k;
  double theta; mw_thetaFn::init(m, n, k);  
  double log_mgf = minimize(mw_thetaFn::minimizationFn, theta);
  log_mgf += theta*k;

  double exp_theta_k[2*m+n+1];
  for(int j = 0; j < 2*m+n+1; j++)
    exp_theta_k[j] = exp(-theta*j);

  //double sigma = mw_thetaFn::get_sigma(-theta, exp_theta_k);
  double sigma = exp(mw_thetaFn::get_log_sigma(-fabs(theta)));
  double lambda = fabs(theta)*sigma;
  
  //
  // Why is this here?? 9/29/06
  //
  //double phi_lambda = integrate_normal(lambda-1/(2*sigma), 0, 1);
  double phi_lambda = integrate_normal(fabs(lambda), 0, 1);
 
  if(DEBUG) {
    cout << "theta: " << theta << endl; 
    cout << "log(MGF): " << log_mgf << endl;
    cout << "sigma: " << sigma << " " << mw_thetaFn::get_sigma(-theta, exp_theta_k) << endl;
    cout << "lambda: " << lambda << endl;
    cout << "Phi: " << phi_lambda << endl;
  }

  double sp = pow(lambda, 2)/2.0 + phi_lambda + log_mgf-theta*k;

  double w3 = (pow(lambda, 2)-1)*exp(-lambda*lambda/2-phi_lambda)/sqrt(2*pi) - pow(lambda, 3);
  double p = m/((double) m+n);
  double c20 = -6*(1-pow(p, 5)-pow(1-p, 5))/(25*pow(p*(1-p), 2));
  double K3 = mw_thetaFn::get_K3(theta);
  double K4 = mw_thetaFn::get_K4(theta);
  double p1 = K3*w3/(6*pow(sigma, 3));
  double p2 = K4*((-pow(lambda, 3)+lambda)*exp(-lambda*lambda/2-phi_lambda)/sqrt(2*pi) + 
		  pow(lambda, 4))/(24*pow(sigma, 4)) +
    pow(K3, 2)*((-pow(lambda, 5)+pow(lambda, 3)-3*lambda)*exp(-lambda*lambda/2-phi_lambda)/sqrt(2*pi) + 
		pow(lambda, 6))/(72*pow(sigma, 6));

  double lam = lambda-1/(2*sigma);
  double phi_ratio = exp(-lam*lam/2-phi_lambda)/sqrt(2*pi);
  double x = -1/(2*sigma);
  double he1 = x; double he2 = -1+pow(x, 2); 
  double he3 = x*(pow(x, 2)-3); 
  double he4 = 3+pow(x, 4)-6*pow(x, 2);
  double he5 = x*(pow(x, 4)-10*pow(x, 2)+15);

  double p1new = K3/(6*pow(sigma, 3))*
    (-pow(lambda, 3) + phi_ratio*
     (pow(lambda, 2)-he1*lambda+he2));
  double p2new = K4/(24*pow(sigma, 4))*
    (pow(lambda, 4) + phi_ratio*
     (-pow(lambda, 3) + he1*pow(lambda, 2) - he2*lambda + he3)) +
     pow(K3, 2)/(72*pow(sigma, 6))*
     (pow(lambda, 6) + phi_ratio*
      (-pow(lambda, 5) + he1*pow(lambda, 4) - he2*pow(lambda, 3) + he3*pow(lambda, 2) - he4*lambda + he5));
    

  double var = m*n*(m+n+1)/12.0;
  double fve_sp = sp + log(1+c20*theta*w3*var*var/(8*pow(sigma, 3)*(m+n)));
  double jr_sp = sp + log(1+p1+p2);
  double jr_sp_new = sp + log(1+p1new+p2new);

  cout << "Values: " << p1 << " " << p2 << " " << K3 << " " << K4 << endl;
  cout << "Results: " << sp << " " << fve_sp << " " << jr_sp << endl;
  cout << c20 << " " << w3 << " " << c20*theta*w3/(8*pow(sigma, 3)*(m+n)) << endl;

  double ROOT2PI = 2.50662827463100;
  double log_B0 = pow(lambda, 2)/2.0 + phi_lambda;    
  double B0 = exp(log_B0);
  double B1 = -lambda*B0+1/ROOT2PI;
  double B2 = -lambda*B1;
  double B3 = -lambda*B2-1/ROOT2PI;  
  double B4 = -lambda*B3;
  double B5 = -lambda*B4+3/ROOT2PI;
  double B6 = -lambda*B5;

  double term2 = B3*K3/(6*pow(sigma, 3));
  double term3 = B4*K4/(24*pow(sigma, 4));
  double term4 = B6*pow(K3, 2)/(72*pow(sigma, 6));

  //double gamma_t = exp(-fabs(theta))/(1-exp(-fabs(theta)));
  double gamma_t = exp(-theta)/(1-exp(-theta));
  double y = (1/theta-gamma_t)/sigma;
  double term5 = y*B1;
  double term6 = gamma_t*(0.5+gamma_t-1/theta)*B2/pow(sigma,2);
  double term7 = y*B4*K3/(6*pow(sigma, 3));

  double jr_sp2 = log(B0 + term2 + term3 + term4) + log_mgf-theta*k;
  double jr_sp3 = log(B0 + term2 + term3 + term4 + term5 + term6 + term7) + log_mgf-theta*k;

  cout << "JR values: " << jr_sp << " " << jr_sp2 << " " << jr_sp3 << endl;
  return fve_sp;
}

#include "extended_exponent.h"

//
// DP in the tied case
//
double* mw_lattice(int m, int n, int k, int c, int* size) {

  ee_t** choose = new ee_t*[m+n+1];
  for(int i = 0; i < m+n+1; i++) {

    choose[i] = new ee_t[m+1];
    choose[i][0].log_set(0);
    for(int j = 1; j < MIN(i+1, m+1); j++)
      choose[i][j] = choose[i][j-1]*ee_t(log((i-j+1)/(j*1.0)));
  }
  
  int Q = 2*m*n+1 + m*(m+1);
  //int Q = k+1 + m*(m+1);
  ee_t** counts = new ee_t*[m+1];
  for(int i = 0; i < m+1; i++) 
    counts[i] = new ee_t[Q];

  int rank = 0;
  for(int d = 0; d < MIN(m+1, size[0]+1); d++)
    counts[d][d*(2*rank+size[0]+1)] = choose[size[0]][d];

  for(int a = 1; a < c; a++) {

    rank += size[a-1];  
    for(int b = MIN(m, rank+size[a]); b >= 0;  b--)
      for(int c = Q-1; c >= 0; c--)
	for(int d = 1; d < MIN(b+1, size[a]+1); d++)
	  if(c >= d*(2*rank+size[a]+1) && counts[b-d][c-d*(2*rank+size[a]+1)] != 0)
	    counts[b][c] += counts[b-d][c-d*(2*rank+size[a]+1)]*choose[size[a]][d];
  }
      
  double* log_pmf = new double[Q-m*(m+1)];
  double log_denominator = choose[m+n][m].log_get();
  for(int i = m*(m+1); i < Q; i++)
    log_pmf[i-m*(m+1)] = counts[m][i].log_get()-log_denominator;

  return log_pmf;
}

//
// LD in the tied case
//
double mw_tied_LD(int m, int n, double k, int c, int* size) {

  ee_t** choose = new ee_t*[m+n+1];
  for(int i = 0; i < m+n+1; i++) {

    choose[i] = new ee_t[m+1];
    choose[i][0].log_set(0);
    for(int j = 1; j < MIN(i+1, m+1); j++)
      choose[i][j] = choose[i][j-1]*ee_t(log((i-j+1)/(j*1.0)));
  }

  k += 1;
  //k = 2*((double) m)*n+m*(m+1)-k;
  double theta; mw_tied_thetaFn::init(m, n, k, c, size, choose);  
  double log_mgf = minimize(mw_tied_thetaFn::minimizationFn, theta);
  log_mgf += theta*k;

  double sigma = mw_tied_thetaFn::get_sigma(theta);
  double lambda = fabs(theta)*sigma;
  //double phi_lambda = integrate_normal(lambda-1/sigma, 0, 1);
  double phi_lambda = integrate_normal(lambda, 0, 1);

  if(DEBUG) {
    cout << "theta: " << theta << endl; 
    cout << "log(MGF): " << log_mgf << endl;
    cout << "sigma: " << sigma << endl;
    cout << "lambda: " << lambda << endl;
    cout << "Phi: " << phi_lambda << endl;
  }

  double sp = pow(lambda, 2)/2.0 + phi_lambda + log_mgf-theta*k;

  double w3 = (pow(lambda, 2)-1)*exp(-lambda*lambda/2-phi_lambda)/sqrt(2*pi) - pow(lambda, 3);
  double p = m/((double) m+n);
  double c20 = -6*(1-pow(p, 5)-pow(1-p, 5))/(25*pow(p*(1-p), 2));
  double K3; double K4;
  mw_tied_thetaFn::get_K(fabs(theta), K3, K4);
  double p1 = -K3*w3/(6*pow(sigma, 3));
  double p2 = K4*((-pow(lambda, 3)+lambda)*exp(-lambda*lambda/2-phi_lambda)/sqrt(2*pi) + 
		  pow(lambda, 4))/(24*pow(sigma, 4)) +
    pow(K3, 2)*((-pow(lambda, 5)+pow(lambda, 3)-3*lambda)*exp(-lambda*lambda/2-phi_lambda)/sqrt(2*pi) + 
		pow(lambda, 6))/(72*pow(sigma, 6));

  double var = m*n*(m+n+1)/12.0;
  double fve_sp = sp + log(1+c20*fabs(theta)*w3*var*var/(8*pow(sigma, 3)*(m+n)));
  double jr_sp = sp + log(1+p1+p2);

  cout << "Values: " << p1 << " " << p2 << " " << K3 << " " << K4 << endl;
  cout << "Results: " << sp << " " << fve_sp << " " << jr_sp << endl;

  return jr_sp;
}

double* mw_tied_sFFT(int m, int n, int k, int c, int* size) {

  ee_t** choose = new ee_t*[m+n+1];
  for(int i = 0; i < m+n+1; i++) {

    choose[i] = new ee_t[m+1];
    choose[i][0].log_set(0);
    for(int j = 1; j < MIN(i+1, m+1); j++)
      choose[i][j] = choose[i][j-1]*ee_t(log((i-j+1)/(j*1.0)));
  }

  double theta; mw_tied_thetaFn::init(m, n, k, c, size, choose);  
  double log_mgf = minimize(mw_tied_thetaFn::minimizationFn, theta);
  log_mgf += theta*k;
  double theta2; minimize(mw_tied_thetaFn::theta2Fn, theta2);
  int Q = 2*m*n + 1 + m*(m+1); 

#if NR
  int Qp = RND_UP_TO_POWER_2(Q);
#else
  int Qp = RND_UP_TO_POWER_2(Q);
  //int Qp = Q;
#endif

  if(DEBUG) {
    cout << "theta: " << theta << endl; 
    cout << "theta2: " << theta2 << endl; 
    cout << "log(MGF): " << log_mgf << endl;
    cout << "Qp: " << Qp << endl;
  }

  double omg = 2*pi/Qp;
  complex *exp_i_omg_x = new complex[Qp];
  for(int x = 0; x < Qp; x++)
    exp_i_omg_x[x].set(cos(omg*x), sin(omg*x));

  int rank = 0; double cum_sum = 0; double cum_sum2 = 0;
  double** w = new double*[c]; double* norm = new double[c];
  double* result = new double[m+1];
  for(int i = 0; i < c; i++) {

    w[i] = new double[MIN(size[i]+1, m+1)]; double sum = LOGZERO;
    for(int j = 0; j < MIN(size[i]+1, m+1); j++)
      sum = log_sum(sum, theta*j*(2*rank+size[i]+1) + 
		      choose[size[i]][j].log_get() + theta2*j);

    cum_sum += sum;
    for(int j = 0; j < MIN(size[i]+1, m+1); j++)
      w[i][j] = exp(theta*j*(2*rank+size[i]+1) + 
		      choose[size[i]][j].log_get() + theta2*j - sum);

    //cout << i << endl;

    if(i == 0) 
      for(int j = 0; j < MIN(size[i]+1, m+1); j++)
	result[j] = w[i][j];
    else {
      conv(w[i], result, result, MIN(size[i]+1, m+1), MIN(rank+size[i]+1, m+1));
      norm[i] = 0;
      for(int j = 0; j < MIN(rank+size[i]+1, m+1); j++)
	norm[i] += result[j];

      for(int j = 0; j < MIN(rank+size[i]+1, m+1); j++)
	result[j] /= norm[i];

      norm[i] = log(norm[i]);
      cum_sum2 += norm[i];
    }

    rank += size[i];
  }

  complex *Dcpmf = new complex[Qp];
  for(int l = 0; l < 1+Qp/2; l++) {

    int rank = 0;
    complex* result = new complex[m+1];
    for(int j = 0; j < MIN(size[0]+1, m+1); j++)
      result[j] = exp_i_omg_x[(l*j*(2*rank+size[0]+1))%Qp]*w[0][j];

    for(int i = 1; i < c; i++) {

      rank += size[i-1];
      complex* a = new complex[MIN(size[i]+1, m+1)];
      for(int j = 0; j < MIN(size[i]+1, m+1); j++)
	a[j] = exp_i_omg_x[(l*j*(2*rank+size[i]+1))%Qp]*w[i][j];

      conv(a, result, result, MIN(size[i]+1, m+1), MIN(rank+size[i]+1, m+1));

      for(int j = 0; j < MIN(rank+size[i]+1, m+1); j++)
	result[j] = result[j].shift(-norm[i]);
    }

    Dcpmf[l] = result[m].shift(cum_sum-theta*m*(m+1)+cum_sum2-log_mgf-choose[m+n][m].log_get() - theta2*m);

    if(l > 0)
      Dcpmf[Qp-l] = Dcpmf[l].conj();
  }

#if NR
  fftnr(Dcpmf, Qp, -1);
#else
  if(Qp < 200000)
    fftw(Dcpmf, Qp, -1);
  else
    fftnr(Dcpmf, Qp, -1);
#endif

  double* log_pmf = new double[Q-m*(m+1)];
  for(int j = m*(m+1); j < Q; j++)
    if(Dcpmf[j].real() > 0)
      log_pmf[j-m*(m+1)] = log(Dcpmf[j].real()) - theta*(j-m*(m+1)) + log_mgf - log((double) Qp);
    else
      log_pmf[j-m*(m+1)] = LOGZERO;

  return log_pmf;
}

#define NC 1

double* mw_tied_csFFT(int m, int n, int k, int c, int* size) {

  ee_t** choose = new ee_t*[m+n+1];
  for(int i = 0; i < m+n+1; i++) {

    choose[i] = new ee_t[m+1];
    choose[i][0].log_set(0);
    for(int j = 1; j < MIN(i+1, m+1); j++)
      choose[i][j] = choose[i][j-1]*ee_t(log((i-j+1)/(j*1.0)));
  }

  double theta; mw_tied_thetaFn::init(m, n, k, c, size, choose);  
  double log_mgf = minimize(mw_tied_thetaFn::minimizationFn, theta);
  log_mgf += theta*k;
  double theta2; minimize(mw_tied_thetaFn::theta2Fn, theta2);
  double sigma = mw_tied_thetaFn::get_sigma(theta);
  double sigma2 = mw_tied_thetaFn::get_sigma2(theta2);
  
  int Q = 2*m*n + 1 + m*(m+1); 

#if NC
  int Qp2 = 2*m+1;
#else
  int Qp2 = (int) MIN(2*m+1, MAX(1, ceil(6*sigma2)));
#endif

#if NR
  int Qp = RND_UP_TO_POWER_2((int) ceil(MIN(Q, 6*sigma)));
#else
  int Qp = RND_UP_TO_POWER_2((int) ceil(MIN(Q, 6*sigma)));
  //int Qp = (int) ceil(MIN(Q, 6*sigma));
#endif

  if(DEBUG) {
    cout << "theta: " << theta << endl; 
    cout << "theta2: " << theta2 << endl; 
    cout << "log(MGF): " << log_mgf << endl;
    cout << "sigma: " << sigma << endl;
    cout << "Sigma2: " << sigma2 << endl;
    cout << "Qp: " << Qp << endl;
    cout << "Qp2: " << Qp2 << endl;
  }

  double omg = 2*pi/Qp;
  complex *exp_i_omg_x = new complex[Qp];
  for(int x = 0; x < Qp; x++)
    exp_i_omg_x[x].set(cos(omg*x), sin(omg*x));

  //theta2 /= 2;

  int rank = 0; double cum_sum = 0; double cum_sum2 = 0;
  double** w = new double*[c]; double* norm = new double[c];
  double* result = new double[m+1];
  for(int i = 0; i < c; i++) {

    w[i] = new double[MIN(size[i]+1, m+1)]; double sum = LOGZERO;
    for(int j = 0; j < MIN(size[i]+1, m+1); j++)
      sum = log_sum(sum, theta*j*(2*rank+size[i]+1) + 
		      choose[size[i]][j].log_get() + theta2*j);

    cum_sum += sum;
    for(int j = 0; j < MIN(size[i]+1, m+1); j++) {
      w[i][j] = exp(theta*j*(2*rank+size[i]+1) + 
		      choose[size[i]][j].log_get() + theta2*j - sum);
      //cout << w[i][j] << " ";
    } //cout << endl;

    if(i == 0) 
      for(int j = 0; j < MIN(size[i]+1, m+1); j++)
	result[j] = w[i][j];
    else {
      conv(w[i], result, result, MIN(size[i]+1, m+1), MIN(rank+size[i]+1, m+1));
      norm[i] = 0;
      for(int j = 0; j < MIN(rank+size[i]+1, m+1); j++)
	norm[i] += result[j];

      for(int j = 0; j < MIN(rank+size[i]+1, m+1); j++) {
	//cout << result[j] << " ";
	result[j] /= norm[i];
      } //cout << endl;

      norm[i] = log(norm[i]);
      cum_sum2 += norm[i];
    }

    rank += size[i];
  }

  complex *Dcpmf = new complex[Qp];
  for(int l = 0; l < 1+Qp/2; l++) {

    int rank = 0;

#if NC    
    complex* result = new complex[m+1];
    //complex* result = new complex[2*m+1];
    for(int j = 0; j < MIN(size[0]+1, m+1); j++)
      result[j] = exp_i_omg_x[(l*j*(2*rank+size[0]+1))%Qp]*w[0][j];
#else
    complex* result = new complex[Qp2];
    for(int j = 0; j < MIN(size[0]+1, m+1); j++)
      result[j%Qp2] += exp_i_omg_x[(l*j*(2*rank+size[0]+1))%Qp]*w[0][j];
#endif

    for(int i = 1; i < c; i++) {

      rank += size[i-1];
#if NC
      complex* a = new complex[MIN(size[i]+1, m+1)];
      //complex* a = new complex[2*m+1];
      for(int j = 0; j < MIN(size[i]+1, m+1); j++)
	a[j] = exp_i_omg_x[(l*j*(2*rank+size[i]+1))%Qp]*w[i][j];

      conv(a, result, result, MIN(size[i]+1, m+1), MIN(rank+size[i]+1, m+1));

      for(int j = MIN(rank+size[i]+1, m+1); j < 2*m+1; j++)
        result[j].set(0, 0);

      for(int j = 0; j < MIN(rank+size[i]+1, m+1); j++)
 	result[j] = result[j].shift(-norm[i]);

#else

      complex* a = new complex[Qp2];
      for(int j = 0; j < MIN(size[i]+1, m+1); j++)
	a[j%Qp2] += exp_i_omg_x[(l*j*(2*rank+size[i]+1))%Qp]*w[i][j];

      conv_fftw(a, result, result, Qp2);

      for(int j = 0; j < Qp2; j++)
	result[j] = result[j].shift(-norm[i]);
#endif
    }

    //Dcpmf[l] = result[m].shift(cum_sum-theta*m*(m+1)+cum_sum2-log_mgf-choose[m+n][m].log_get() - theta2*m);
    Dcpmf[l] = result[m%Qp2].shift(cum_sum-theta*m*(m+1)+cum_sum2-log_mgf-choose[m+n][m].log_get() - theta2*m);

    if(l > 0)
      Dcpmf[Qp-l] = Dcpmf[l].conj();
  }

#if NR
  fftnr(Dcpmf, Qp, -1);
#else
  if(Qp < 200000)
    fftw(Dcpmf, Qp, -1);
  else
    fftnr(Dcpmf, Qp, -1);
#endif

  double* log_pmf = new double[Q-m*(m+1)];
  for(int j = 0; j < Q-m*(m+1); j++) log_pmf[j] = LOGZERO;

  for(int j = MAX(0, (int) k+m*(m+1)-Qp/2); j <= k+m*(m+1); j++)
    if(Dcpmf[j%Qp].real() > 0) {
      log_pmf[j-m*(m+1)] = log(Dcpmf[j%Qp].real()) - theta*(j-m*(m+1)) + log_mgf -log((double) Qp);
      //cout << j-m*(m+1) << " " << log_pmf[j-m*(m+1)] << endl;
    }

  return log_pmf;
}
