#ifndef _MOTIF_EVALUATOR_H_
#define _MOTIF_EVALUATOR_H_

#include <vector>
using namespace std;

const double log_acc_level = 2*log(10);

#define FIXED 0

class motif_evaluator {

 private:
  int maxN; int N; int K; int minL; int maxL; 
  double* pu; int Q;
  double step; double** log_pmf; 
  double** log_pvalues; double** log_acc;

  double mean;
  void precompute(double theta, double s = -1);

 public:

  double** Iupd; double** pairUpd;

  motif_evaluator() {}

  void init(int given_maxN, int given_K, int given_minL, int given_maxL, 
	    double* given_pu, int given_Q); 

  void setN(int given_N);
  int getN() { return N; }

  
  double get_pvalue_careful(int L_minus_min, double s) {

    int I = int(s/step);

    if(log_acc[L_minus_min][I] > log_acc_level)
      return log_pvalues[L_minus_min][I]; 
    else {
      
      precompute(0, s/((double) L_minus_min+minL));
      return log_pvalues[L_minus_min][I]; 
    }
  }

  double get_pvalue(int L_minus_min, double s) {

#if FIXED
    //static double* log_values = log_pvalues[0];
    return log_pvalues[0][int(s/step)];
    //return log_values[int(s/step)];
#else
    return log_pvalues[L_minus_min][int(s/step)];
    //return get_pvalue_careful(L_minus_min, s);
#endif
  }

  ~motif_evaluator();
};

#endif
