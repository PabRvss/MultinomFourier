#include <iostream>
#include <math.h>
#include "utils.h"

using namespace std;

//
// Returns data[min_index] where data[min_index] = min(data[0..(size-1)])
//  
double min_array(double* data, int size, int& min_index) {

  double min_value = data[0]; min_index = 0;
  for(int i = 1; i < size; i++)
    if(min_value > data[i]) {
      min_value = data[i]; min_index = i;
    }

  return min_value;
}

//
// Sets out[0..(row_size-1)][0..(column_size-1)] to
// exp(data[0..(row_size-1)][0..(column_size-1)])
//
void exp_2Darray(double** data, int row_size, int column_size, double** out) {

  for(int i = 0; i < row_size; i++)
    for(int j = 0; j < column_size; j++)
      out[i][j] = exp(data[i][j]);
}

//
// Sets out[0..(row_size-1)][0..(column_size-1)] to
// exp(data[0..(row_size-1)][0..(column_size-1)])
//
void exp_2Darray(double** data, int row_size, int column_size, ee_t** out) {

  for(int i = 0; i < row_size; i++)
    for(int j = 0; j < column_size; j++)
      out[i][j].log_set(data[i][j]);
}

//
// Computes log_sum for the entire array of size n
//    
double log_array_sum(double* log_x, int n) {

  double array_sum = log_x[0];
  for(int i = 1; i < n; i++)
    array_sum = log_sum(array_sum, log_x[i]);

  return array_sum;
}

//
// Computes the L1 norm of the vector x of size n
// 
double one_norm(double* x, int n) {

  double sum = 0;
  for(int i = 0; i < n; i++)
    sum += fabs(x[i]);

  return sum;
}

//
// Computes the L2 norm of the vector x of size n
// 
double two_norm(double* x, int n) {

  double sum = 0;
  for(int i = 0; i < n; i++)
    sum += x[i]*x[i];

  return sqrt(sum);
}

//
// Reads the parameter list for a program that computes
// the p-value for the multinomial llr statistic
//
void read_llr_parameters(int argc, char** argv, int& N, int& K, double& s, int& Q, double*& pu) {

  if(argc < 5) {
    cout << "Computes the p-value for the multinomial llr statistic" << endl << endl
	 << "Usage: <program name> N K s Q pu" << endl
	 << "N = Number of objects" << endl
	 << "K = Number of bins" << endl
	 << "s = LLR score" << endl
	 << "Q = Lattice size" << endl
	 << "pu = List of background probabilities (of size K)" << endl << endl
	 << "Example: <program name> 50 4 50 16384 0.25 0.25 0.25 0.25" << endl;
    exit(1);
  }

  int argno = 1;
  N = atoi(argv[argno++]); K = atoi(argv[argno++]); 
  s = atof(argv[argno++]); Q = atoi(argv[argno++]); 
  pu = new double[K];

  cout << "Parameters: N = " << N << " K = " << K << " s = " << s << " Q = " << Q << " pu =";
  for(int i = 0; i < K; i++) { pu[i] = atof(argv[argno++]); cout << " " << pu[i]; }
  cout << endl;
}

//
// Reads the parameter list for a program that computes
// the p-value for the entropy score
//
void read_entropy_parameters(int argc, char** argv, int& N, int& K, int& L, double& s, int& Q, double*& pu) {

  if(argc < 6) {
    cout << "Computes the p-value for the entropy score" << endl << endl
	 << "Usage: <program name> N K L s Q pu" << endl
	 << "N = Number of objects" << endl
	 << "K = Number of bins" << endl
	 << "L = Number of i.i.d. samples" << endl
	 << "s = Entropy score" << endl
	 << "Q = Lattice size" << endl
	 << "pu = List of background probabilities (of size K)" << endl << endl
	 << "Example: <program name> 50 4 10 500 16384 0.25 0.25 0.25 0.25" << endl;
    exit(1);
  }

  int argno = 1;
  N = atoi(argv[argno++]); K = atoi(argv[argno++]); L = atoi(argv[argno++]); 
  s = atof(argv[argno++]); Q = atoi(argv[argno++]); 
  pu = new double[K];

  if(argc < 6+K) {
    cout << "Too few arguments" << endl;
    exit(1);
  }

  cout << "Parameters: N = " << N << " K = " << K << " L = " << L << " s = " << s << " Q = " << Q << " pu =";
  for(int i = 0; i < K; i++) { pu[i] = atof(argv[argno++]); cout << " " << pu[i]; }
  cout << endl;

  //llr_score_t score(N, K, pu, Q);
  //if(s < L*score.get_min_ent() || s > L*score.get_max_ent())
  //  exit(1);
  //cout << L*score.get_min_ent() << endl;
  //if(s < L*score.get_min_ent())
  //s = L*score.get_min_ent();
}

//
// Computes the error bounds for the given score s and theta
//
void compute_theta_err_bnds(double* log_err_bnd, double step, int Q, double s, int K, 
			    double& log_min_err_bnd, double& log_max_err_bnd, double theta) {

  int i;   
  log_max_err_bnd = LOGZERO; log_min_err_bnd = LOGZERO;

  for(i = (int) MAX(0, (s/step)-(K/2.0)); i < (int) MIN((s/step)+(K/2.0)+1, Q); i++)
    log_max_err_bnd = log_sum(log_max_err_bnd, log_err_bnd[i]);

  if(theta > 0)
    for(; i < Q; i++)
      log_min_err_bnd = log_sum(log_min_err_bnd, log_err_bnd[i]);
  else
    for(i = (int) MAX(0, (s/step)-(K/2))-1; i >= 0; i--)
      log_min_err_bnd = log_sum(log_min_err_bnd, log_err_bnd[i]);

  log_max_err_bnd = log_sum(log_max_err_bnd, log_min_err_bnd);
}

//
// Computes the p-value bounds for the given score s
//
void compute_pvalues(double* log_pmf, double step, int Q, double s, int K, 
		     double& log_min_pval, double& log_max_pval) {

  int i;   
  log_max_pval = LOGZERO; log_min_pval = LOGZERO;

  for(i = (int) MAX(0, s/step - K/2.0); i < (int) MIN(s/step + K/2.0 + 1, Q); i++)
    log_max_pval = log_sum(log_max_pval, log_pmf[i]);

  for(; i < Q; i++)
    log_min_pval = log_sum(log_min_pval, log_pmf[i]);

  log_max_pval = log_sum(log_max_pval, log_min_pval);
}

//
// Computes the p-value bounds for the given score s and theta
//
void compute_theta_sum(double* log_pmf, double step, int Q, double s, int K, 
		     double& log_min_pval, double& log_max_pval, double theta, int invert) {
  
  if(theta > 0)
    compute_pvalues(log_pmf, step, Q, s, K, log_min_pval, log_max_pval);
  else {

    int i;   
    double log_max_pval_inv = LOGZERO; double log_min_pval_inv = LOGZERO;

    for(i = (int) MAX(0, s/step - K/2.0); i < (int) MIN(s/step + K/2.0 +1, Q); i++)
      log_min_pval_inv = log_sum(log_min_pval_inv, log_pmf[i]);
    
    for(i = 0; i < (int) MAX(0, s/step - K/2.0); i++)
      log_max_pval_inv = log_sum(log_max_pval_inv, log_pmf[i]);
    
    log_min_pval_inv = log_sum(log_min_pval_inv, log_max_pval_inv);

    log_max_pval = (invert ? log(1-exp(log_max_pval_inv)) : log_max_pval_inv);
    log_min_pval = (invert ? log(1-exp(log_min_pval_inv)) : log_min_pval_inv);
  }
}

void compute_theta_pvalues(double* log_pmf, double step, int Q, double s, int K, 
		     double& log_min_pval, double& log_max_pval, double theta) {

  compute_theta_sum(log_pmf, step, Q, s, K, log_min_pval, log_max_pval, theta, 1);
}

void compute_theta_bounds(double* log_pmf, double step, int Q, double s, int K, 
		     double& log_min_pval, double& log_max_pval, double theta) {

  compute_theta_sum(log_pmf, step, Q, s, K, log_min_pval, log_max_pval, theta, 0);
}

complex_t recursive_sum(complex_t* data, int size) {

  if(size == 2) 
    return data[0]+data[1];
  else 
    return recursive_sum(data, size/2)+recursive_sum(data+size/2, size/2);
}

void compute_phi_pvalue(complex_t* phi, double step, int size, double start, int end, int uncertainity, 
			double& log_min_pval, double& log_max_pval, double theta, double log_mgf) {

  complex_t phi_max_sum;
  complex_t phi_min_sum;

  double omg = 2*pi/size;
  int j_upper = (int) MAX(0, start/step - uncertainity/2.0);
  int j_lower = (int) MIN(start/step + uncertainity/2.0 + 1, size);

  //complex_t* phi_max = new complex_t[size];
  //complex_t* phi_min = new complex_t[size];

  for(int It = 0; It < size; It++) {
    complex_t x(-theta*step, -omg*It);
    complex_t x_Qju(-theta*step*(end-j_upper), sign(j_upper-end)*omg*mod(abs(end-j_upper), It, size));
    complex_t x_Qjl(-theta*step*(end-j_lower), sign(j_lower-end)*omg*mod(abs(end-j_lower), It, size));
    complex_t x_ju(0, -omg*mod(j_upper, It, size));
    complex_t x_jl(0, -omg*mod(j_lower, It, size));

    phi_max_sum += phi[It]*exp(x_ju)*exp_minus_one(x_Qju)/exp_minus_one(x);  
    phi_min_sum += phi[It]*exp(x_jl)*exp_minus_one(x_Qjl)/exp_minus_one(x); 
    //phi_max[It] = phi[It]*exp(x_ju)*exp_minus_one(x_Qju)/exp_minus_one(x);  
    //phi_min[It] = phi[It]*exp(x_jl)*exp_minus_one(x_Qjl)/exp_minus_one(x); 
  }

  //phi_max_sum = recursive_sum(phi_max, size);
  //phi_min_sum = recursive_sum(phi_min, size);

  log_min_pval = (phi_min_sum.real() != 0 ?  log(fabs(phi_min_sum.real())) - theta*step*j_lower + log_mgf - log(size) :
		  LOGZERO);
  log_max_pval = (phi_max_sum.real() != 0 ?  log(fabs(phi_max_sum.real())) - theta*step*j_upper + log_mgf - log(size) :
		  LOGZERO);
}

