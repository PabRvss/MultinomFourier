#include "utils.h"
#include "naive_enumeration.cpp"
#include "convolution.h"
#include "motif_evaluator.h"

void motif_evaluator::init(int given_maxN, int given_K, int given_minL, int given_maxL, 
			   double* given_pu, int given_Q) {

  maxN = given_maxN; K = given_K; minL = given_minL; maxL = given_maxL; 
  pu = given_pu; Q = given_Q;
  log_pmf = multi_naive_enumeration(maxN, pu, Q, step);

  log_pvalues = new double*[maxL-minL+1];
  log_acc = new double*[maxL-minL+1];
  for(int L = minL; L <= maxL; L++) {
    log_pvalues[L-minL] = new double[L*Q];
    log_acc[L-minL] = new double[L*Q];
  }

  Iupd = new double*[K]; 
  for(int i = 0; i < K; i++) {

    Iupd[i] = new double[maxN]; Iupd[i][0] = log(1/pu[i]); 
    for(int j = 1; j < maxN; j++)
      Iupd[i][j] = (j+1)*log((j+1)/pu[i])-j*log(j/pu[i]);
  }

  pairUpd = new double*[K];
  for(int i = 0; i < K; i++) {
     pairUpd[i] = new double[K];
     for(int j = 0; j < K; j++)
       pairUpd[i][j] = log(1/pu[i])+log(1/pu[j])-2*log(2);
     pairUpd[i][i] = 2*log(1/pu[i]);
  }
}

#include "minimize.h"
class thetaFn {

private:

  static llr_score_t* score;
  static multinom_prob_t<ee_t>* prob;
  static int N; static double st;

public:

  static void init(int given_N, int K, double* pu, int Q, double given_st) {
  
    N = given_N; st = given_st;
    score = new llr_score_t(N, K, pu, Q);
    prob = new multinom_prob_t<ee_t>(N, K, pu);
    set_ee_p(*prob, *score);
  }

  static double minimizationFn(double theta) {

    ee_t sum;
    for(int i = 0; i <= N; i++)
      for(int j = 0; j <= N-i; j++)
	for(int k = 0; k <= N-(i+j); k++) {
	  int l = N-(i+j+k);
	  double I = score->get_I(0, i) + score->get_I(1, j) +
	    score->get_I(2, k) + score->get_I(3, l);
	  ee_t p = prob->get_p(0, i)*prob->get_p(1, j)*
	    prob->get_p(2, k)*prob->get_p(3, l);
	  sum += p*ee_t(theta*I);
	}
 
    return sum.log_get()+prob->log_const_factor() - theta*st;
  }

  static void clear() {

    delete(score); delete(prob);
  }
};

llr_score_t* thetaFn::score;
multinom_prob_t<ee_t>* thetaFn::prob;
int thetaFn::N; double thetaFn::st;
 
void motif_evaluator::precompute(double theta, double s) {

  double* curr_log_pmf = log_pmf[N];
  double log_mgf = LOGZERO; 

  //int curr_Q = MIN(Q, NINT(Q/((double) maxN)*N));
  int curr_Q = MIN(Q, int(Q/((double) maxN)*N));
  int period = RND_UP_TO_POWER_2(maxL*curr_Q);

  if(s != -1) {
    thetaFn::init(N, K, pu, Q, s);
    log_mgf = minimize(thetaFn::minimizationFn, theta);
    log_mgf += theta*s;
    cout << "Precomputed with: " << theta << " " << s << endl;
    thetaFn::clear();
  }

  double* pmf = new double[period]; mean = 0;
  for(int It = 0; It < period; It++)
    pmf[It] = 0;

  for(int It = 0; It < curr_Q; It++) {
    pmf[It] = curr_log_pmf[It] + theta*step*It;
    mean += It*exp(curr_log_pmf[It]);
  }

  if(log_mgf == LOGZERO)
    log_mgf = log_array_sum(pmf, curr_Q);

  for(int It = 0; It < curr_Q; It++)
    pmf[It] = exp(pmf[It] - log_mgf);

#if NR
  double** pmfL = multi_self_conv_fftnr(pmf, minL, maxL, period);
#else
  //double** pmfL = multi_self_conv_fftw(pmf, minL, maxL, period);
  double** pmfL = new double*[maxL-minL+1];
  for(int L = minL; L <= maxL; L++)
  pmfL[L-minL]= self_conv_fftw(pmf, curr_Q, L, RND_UP_TO_POWER_2(L*curr_Q));
  //pmfL[L-minL]= self_conv_fftw(pmf, period, L, period);
#endif

  delete[](pmf);

  for(int L = minL; L <= maxL; L++) {
    
    int It = L*curr_Q-1; 
    double log_err_term = log(EPS*L*(20*log(period)+20)) + L*log_mgf;

    double log_err_sum = LOGZERO; double log_pvalue_sum = LOGZERO;
    
    /*
    if(pmfL[L-minL][It] > 0) {
      log_err_sum = -theta*step*It;
      log_pvalue_sum = log(pmfL[L-minL][It]) - theta*step*It + L*log_mgf;
    }

    if(log_pvalue_sum - (log_err_sum + log_err_term) > log_acc[L-minL][It]) {
      log_acc[L-minL][It] = log_pvalue_sum - (log_err_sum + log_err_term); 
      log_pvalues[L-minL][It] = log_pvalue_sum;
    }
    */

    log_pvalues[L-minL][It] = log_pvalue_sum = -It*step;

    for(int i = It+1; i < L*Q; i++)
      log_pvalues[L-minL][i] = log_pvalues[L-minL][It];

    for(It--; It >= 0; It--) {

      if(pmfL[L-minL][It] > 0) {
	log_pvalue_sum = log_sum(log_pvalue_sum, log(pmfL[L-minL][It]) - theta*step*It + L*log_mgf);       
	log_err_sum = log_sum(log_err_sum, -theta*step*It);
      }

      //if(log_pvalue_sum - (log_err_sum + log_err_term) > log_acc[L-minL][It]) {
	log_acc[L-minL][It] = log_pvalue_sum - (log_err_sum + log_err_term); 
	log_pvalues[L-minL][It] = log_pvalue_sum;
	//}
    }

    delete[](pmfL[L-minL]);
  }

  delete[](pmfL);
}

void motif_evaluator::setN(int given_N) {

  N = given_N;
  //int curr_Q = MIN(Q, NINT(Q/((double)maxN)*N));

  for(int L = minL; L <= maxL; L++)
    //for(int It = 0; It < L*curr_Q; It++) {
    for(int It = 0; It < L*Q; It++) {
      log_pvalues[L-minL][It] = 0;
      log_acc[L-minL][It] = -100;
    }

  precompute(1);
}

motif_evaluator::~motif_evaluator() {

  for(int i = 0; i < maxN+1; i++)
    delete[](log_pmf[i]);

  for(int i = minL; i <= maxL; i++)
    delete[](log_pvalues[i-minL]);

  for(int i = 0; i < K; i++) {
    delete[](Iupd[i]); delete[](pairUpd[i]);
  }

  delete[](Iupd); delete[](pairUpd);
  delete[](log_pmf); delete[](log_pvalues);
}

