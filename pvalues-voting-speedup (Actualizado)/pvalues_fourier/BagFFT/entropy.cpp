#include <iostream>
#include <math.h>
#include "utils.h"
#include "theta_fns.h"
#include "minimize.h"
#include "convolution.h"
#include "extended_exponent.h"
#include "list.h"
#include "entropy.h"

using namespace std;

//
// Function: Computes the lattice pmf (log-values in an array of size
//           L*Q) for a sum of scores based on the NC algorithm.
//
// Parameters: log_pmf = Log of lattice pmf for single sample
//             L = Number of i.i.d. samples
//             Q = Lattice size for single sample
//
double* NC(double* log_pmf, int Q, int L) {

  ee_t pmf[Q];
  for(int i = 0; i < Q; i++)
    pmf[i].log_set(log_pmf[i]);

  ee_t* pmfL = self_conv(pmf, Q, L);

  double* log_pmfL = new double[L*Q];
  for(int i = 0; i < L*Q; i++)
    log_pmfL[i] = pmfL[i].log_get();

  return log_pmfL;
}

//
// Function: Computes the p-value for a sum of scores based on the 
//           NC algorithm with pruning and lists.
//
// Parameters: log_pmf = Log of lattice pmf for single sample
//             L = Number of i.i.d. samples
//             Q = Lattice size for single sample
//             s = Score to compute p-value for
//
double* NCpruned(double* log_pmf, int Q, int L, double s) {

  //
  // Note: need to add the pruning and return a double!!
  //

  //
  // Initializing the extended exponent lists
  //
  list_t<ee_t> pmf(Q); list_t<ee_t> result(Q);
  for(int i = 0; i < Q; i++) {
    if(log_pmf[i] > LOGZERO) {
      pmf.set(i, ee_t(log_pmf[i]));
      result.set(i, ee_t(log_pmf[i]));
    }
  }

  //
  // Doing the L-fold convolution
  //
  for(int l = 2; l <= L; l++) {

    list_t<ee_t> new_result(l*Q);
    for(int i = pmf.start(); pmf.not_last(); i = pmf.next())
      for(int j = result.start(); result.not_last(); j = result.next())
	new_result.set(i+j, new_result[i+j] + pmf[i]*result[j]);

    result.destructive_copy(new_result);
  }

  double* log_pmfL = new double[L*Q];
  for(int i = 0; i < L*Q; i++)
    log_pmfL[i] = LOGZERO;

  for(int i = result.start(); result.not_last(); i = result.next())
    log_pmfL[i] = log(result[i]);

  return log_pmfL;
}

//
// Function: Computes the lattice pmf (log-values in an array of size
//           L*Q) for a sum of scores based on the sFFT algorithm.
//
// Parameters: log_pmf = Log of lattice pmf for single sample
//             L = Number of i.i.d. samples
//             s = Score for which theta is optimized
//             Q = Lattice size for single sample
//             step = Step size for the lattice
//             theta = Optimal theta shift
//             log_mgf = Log of the mgf of the pmf at theta
//
// If theta is set to NOT_COMPUTED then it is computed and otherwise
// the given theta is used.
//
double* sFFT(double* log_pmf, int L, double s, int Q, double step, double& theta, double& log_mgf) {

  //
  // Computing theta
  //
  if(theta == NOT_COMPUTED) {
    thetaFn::init(log_pmf, step, s/((double) L), Q);
    log_mgf = minimize(thetaFn::minimizationFn, theta);
    log_mgf += thetaFn::get_s()*theta;
    if(DEBUG)
      cout << "theta: " << theta << endl
	   << "log(MGF): " << log_mgf << endl;
  }

  //
  // Shifting the pmf
  //
  double* pmf = log_pmf;
  for(int It = 0; It < Q; It++)
    pmf[It] = exp(log_pmf[It] + theta*step*It - log_mgf);

  //
  // Computing the convolution of the shifted pmf
  //
#if NR
  int period = RND_UP_TO_POWER_2(L*Q);
  double* pmfL = self_conv_fftnr(pmf, Q, L, period);
#else
  int period = L*Q;
  double* pmfL = self_conv_fftw(pmf, Q, L, period);
#endif

  //
  // Unshifting the result
  //
  double* log_pmfL = pmfL;
  for(int It = 0; It < period; It++)
    log_pmfL[It] = (pmfL[It] > 0 ? log(pmfL[It]) : LOGZERO) -
      theta*step*It + L*log_mgf;

  return log_pmfL;
}

const int ksigma = 4;

//
// Function: Computes the lattice pmf (log-values in an array of size
//           L*Q) for a sum of scores based on the csFFT algorithm.
//
// Parameters: log_pmf = Log of lattice pmf for single sample
//             L = Number of i.i.d. samples
//             s = Score for which theta is optimized
//             Q = Lattice size for single sample
//             step = Step size for the lattice
//             theta = Optimal theta shift
//             log_mgf = Log of the mgf of the pmf at theta
//
// If theta is set to NOT_COMPUTED then it is computed and otherwise
// the given theta is used.
//
double* csFFT(double* log_pmf, int L, double s, int Q, double step, double& theta, double& log_mgf) {

  //
  // Computing theta
  //
  thetaFn::init(log_pmf, step, s/((double) L), Q);
  if(theta == NOT_COMPUTED) {
    log_mgf = minimize(thetaFn::minimizationFn, theta);
    log_mgf += thetaFn::get_s()*theta;
    if(DEBUG)
      cout << "theta: " << theta << endl
	   << "log(MGF): " << log_mgf << endl;
  }

  //
  // Computing the period
  //
  double sigma = thetaFn::get_sigma(theta);
  int Lprime = (int) ceil(ksigma*sigma*sqrt((double) L)/(step*Q));
  Lprime = MIN(L, MAX(1, Lprime));

  if(DEBUG)
    cout << "sigma: " << sigma << endl
	 << "Lprime: " << Lprime << endl;

  //
  // Boosting (Not implemented: boosting for the left tail)
  //
  if(theta > 0 && L*Q*step-s+L*100 <= Lprime*Q*step/2.0) {

    theta += 20.0/(L*Q*step-s);
    sigma = thetaFn::get_sigma(theta);
    Lprime = (int) MAX(1, ceil(ksigma*sigma*sqrt((double) L)/(step*Q)));
    //Lprime = (int) ceil(L-s/(Q*step));
    Lprime = MIN(L, MAX(1, Lprime));
    log_mgf = thetaFn::minimizationFn(theta) + thetaFn::get_s()*theta;

    if(DEBUG) 
      cout << "Boosted theta: " << theta << endl
	   << "Boosted sigma: " << sigma << endl
	   << "Boosted log(MGF): " << log_mgf << endl
	   << "Boosted Lprime: " << Lprime << endl;
  }
 
  //
  // Shifting the pmf
  //
  double* pmf = log_pmf;
  for(int It = 0; It < Q; It++)
    pmf[It] = exp(log_pmf[It] + theta*step*It - log_mgf);

  //
  // Computing the convolution of the shifted pmf
  //  
#if NR
  int period = RND_UP_TO_POWER_2(Lprime*Q);
  double* pmfL = self_conv_fftnr(pmf, Q, L, period);
#else
  int period = Lprime*Q;
  double* pmfL = self_conv_fftw(pmf, Q, L, period);
#endif

  //
  // Unshifting the result
  //
  double* log_pmfL = new double[L*Q];
  for(int It = 0; It < L*Q; It++)
    if(It*step > s-(period/2)*step && It*step < s+(period/2)*step)
      log_pmfL[It] = (pmfL[It%period] > 0 ? log(pmfL[It%period]) : LOGZERO) -
	theta*step*It + L*log_mgf;
    else
      log_pmfL[It] = LOGZERO;

  return log_pmfL;
}
