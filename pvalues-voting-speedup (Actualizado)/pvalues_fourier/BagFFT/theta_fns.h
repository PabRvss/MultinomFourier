#ifndef _THETA_OBJS_H_
#define _THETA_OBJS_H_

#include "extended_exponent.h"
#include "log_value.h"
#include "convolution.h"
#include "utils.h"

//
// Namespace for the variables and functions
// to use to compute theta1 in the bagFFT algorithm
//
template<class score_t, class prob_t>
class theta1Fn {
  
 private:

  static score_t* score; static prob_t* prob;

 public:

  static double s;

  //
  // To initialize the namespace
  //
  static void init(score_t* given_score, prob_t* given_prob, double given_s) { 

    score = given_score; prob = given_prob; s = given_s;
  }

  //
  // Returns -theta*s + log(M(theta))
  //
  static double minimizationFn(double theta) {

    int const N = prob->get_N(); double step = score->get_step();
    ee_t * op1 = new ee_t[N+1]; 
    ee_t * op2 = new ee_t[N+1]; 
 
    //
    // log(M(theta)) is computed by
    // convolving the shifted p[k]'s in extended exponent
    // arithmetic 
    //
    for(int i = 0; i < N+1; i++)
      op1[i].log_set(prob->get_log_p(0, i) + 
		     theta*step*score->get_Ip(0, i));

    for(int k = 1; k < prob->get_K(); k++) {
      
      for(int i = 0; i < N+1; i++)      
	op2[i].log_set(prob->get_log_p(k, i) + 
		       theta*step*score->get_Ip(k, i));
  
      conv(op2, op1, op1, N+1, N+1);
    }

    double result = -s*theta + op1[N].log_get() + prob->log_const_factor();

    // Clean up
    delete [] op1;
    delete [] op2;

    return result;
  }

  //
  // To compute sigma for the given theta and offset
  // The offset should be s.t. offset + step*Ip[k][n] > 0
  //
  //
  static double get_sigma(double theta, double offset) {
    
    int const N = prob->get_N(); int const K = prob->get_K();
    double step = score->get_step();
    int** Ip = score->get_Ip_copy(); double** log_p = prob->get_log_p_copy();
    
    //
    // p0 = Shifted p for given prob
    // p1 = first derivative of p0
    // p2 = second derivative of p0
    //
    ee_t** p0 = new ee_t*[K]; ee_t** p1 = new ee_t*[K]; ee_t** p2 = new ee_t*[K];
    for(int k = 0; k < K; k++) {
      p0[k] = new ee_t[N+1]; p1[k] = new ee_t[N+1]; p2[k] = new ee_t[N+1];
    }

    for(int k = 0; k < K; k++)
      for(int n = 0; n < N+1; n++) {
	
	//
	// The offset is applied to ensure that we don't
	// take log of negative values. After
	// we compute the moments we will have to correct
	// for the offset.
	//
	double s = step*Ip[k][n]+offset;
	p0[k][n].log_set(theta*s + log_p[k][n]); 
      	p1[k][n].log_set(theta*s + log_p[k][n] + log(s));
	p2[k][n].log_set(theta*s + log_p[k][n] + 2*log(s)); 
      }


    ee_t* out1 = new ee_t[N+1]; ee_t* out2 = new ee_t[N+1]; ee_t two(log(2.0));

    //
    // m0 = M(theta)
    // m1 = M'(theta)
    // m2 = M''(theta)
    //
    ee_t* m0 = p0[0]; ee_t* m1 = p1[0]; ee_t* m2 = p2[0];
    for(int k = 1; k < K; k++) {

      conv(p0[k], m2, m2, N+1, N+1);
      conv(p1[k], m1, out1, N+1, N+1);
      conv(p2[k], m0, out2, N+1, N+1);

      for(int n = 0; n < N+1; n++)
	m2[n] += out2[n] + two*out1[n];

      conv(p0[k], m1, m1, N+1, N+1);
      conv(p1[k], m0, out1, N+1, N+1);
 
      for(int n = 0; n < N+1; n++) 
	m1[n] += out1[n];

      conv(p0[k], m0, m0, N+1, N+1);
    }

    //
    // m1N = Normalized first moment
    // m2N = Normalized second moment
    // Correcting for the offset.
    //
    double m1N = exp(m1[N].log_get()-m0[N].log_get()) - K*offset;
    double m2N = exp(m2[N].log_get()-m0[N].log_get()) - 2*K*offset*m1N - pow(K*offset, 2);

    // Clean up
    for(int k = 0; k < K; k++) {
      delete[] p0[k]; delete[] p1[k]; delete[] p2[k];
    }
    delete[] p0; delete[] p1; delete[] p2;
    delete[] out1; delete[] out2;

    return sqrt(m2N-m1N*m1N);
  }
};

template<class score_t, class prob_t> score_t* theta1Fn<score_t, prob_t>::score; 
template<class score_t, class prob_t> prob_t* theta1Fn<score_t, prob_t>::prob;
template<class score_t, class prob_t> double theta1Fn<score_t, prob_t>::s;  

//
// Namespace for the variables and functions
// to use to compute theta2 in the bagFFT algorithm
//
template<class prob_t>
class theta2Fn {

 private:

  static prob_t* prob;

  //
  // FFT array size
  //
  static int N2;

  static int** lat;
  static double step;
  static double* pu;

 public:

  //
  // To initialize the namespace
  //  
  static void init(prob_t* given_prob, int given_N2, int** g, double s, double* x) 
    { prob = given_prob; N2 = given_N2; lat = g; step = s; pu = x; }    

  //
  // Sets out[i] to exp(log_in[i]-theta*i)/C(theta) where
  // C(theta) is such that out sums to 1. Returns log(C(theta)).
  //
  static double shift_and_normalize(double* log_in, int size, double theta, double* out) {

    for(int i = 0; i < size; i++)
      out[i] = log_in[i] - theta*i; 
    
    double log_sum = log_array_sum(out, size);
    
    for(int i = 0; i < size; i++)
      out[i] = exp(out[i] - log_sum);
    
    return log_sum;
  }

  //
  // Returns the error bound for doing convolutions with
  // theta as described in the bagFFT paper
  //
  static double minimizationFn(double theta) {

    int const N = prob->get_N();
    double* op1 = new double[N2]; double* op2 = new double[N2];
    for(int i = 0; i < N2; i++)
      op1[i] = op2[i] = 0;

    double** log_p = prob->get_log_p_copy();

    double total_log_sum = shift_and_normalize(log_p[0], N+1, theta, op1);
    double* op3 = new double[N+1];
    for(int i = 0; i < N+1; i++)
      op3[i] = op1[i]*(1 + (i > 0 ? log((double) i): i) + 
      	       fabs((prob->get_theta1()-1)*lat[0][i]*step) + 
      	       fabs((theta-1)*i));
      //op3[i] = op1[i]*(1 + N*pu[0] + (i > 0 ? i*(log((double) i)-1): i) + 
      //	       fabs(prob->get_theta1()*lat[0][i]*step) + 
      //	       i*(theta+log(N*pu[0])));
    double err = two_norm(op3, N+1);
  
    for(int k = 1; k < prob->get_K(); k++) {
    
      total_log_sum += shift_and_normalize(log_p[k], N+1, theta, op2);
      for(int i = 0; i < N+1; i++)
	op3[i] = op2[i]*(1 + (i > 0 ? log((double) i): i) + 
			 fabs((prob->get_theta1()-1)*lat[k][i]*step) + 
			 fabs((theta-1)*i));
      //op3[i] = op2[i]*(1 + N*pu[k] + (i > 0 ? i*(log((double) i)-1): i) + 
      //		 fabs(prob->get_theta1()*lat[k][i]*step) + 
      //		 i*(theta+log(N*pu[k])));
      double mx = two_norm(op3, N+1);
      err += two_norm(op1, N+1)*(2*10*log((double) N2)/log(2.0) + 5) +
	//	one_norm(op1, N+1)*two_norm(op2, N+1)*(log((double) N2)+1);
	one_norm(op1, N+1)*(two_norm(op2, N+1)*10*log((double) N2)/log(2.0) + mx);

#if NR
      real_conv_fftnr(op2, op1, op1, N2);
#else
      real_conv_fftw(op2, op1, op1, N2);
#endif

      for(int n = N+1; n < N2; n++) op1[n] = 0;
    }

    delete[] op1; delete[] op2; delete[] op3;

    return log(err*EPS) + theta*N + total_log_sum;
  }

  //
  // Returns the error bound for doing convolutions with
  // multiple thetas as described in the bagFFT paper
  //
  static double error_bound(double* theta) {

    int const N = prob->get_N();
    double* op1 = new double[N2]; double* op2 = new double[N2];
    for(int i = 0; i < N2; i++)
      op1[i] = op2[i] = 0;

    double** log_p = prob->get_log_p_copy();

    double total_log_sum = shift_and_normalize(log_p[0], N+1, theta[0], op1);
    double err = two_norm(op1, N+1);
  
    for(int k = 1; k < prob->get_K(); k++) {
    
      total_log_sum += shift_and_normalize(log_p[k], N+1, theta[k], op2);
      err += two_norm(op1, N+1)*(2*log((double) N2)+8) +
	one_norm(op1, N+1)*two_norm(op2, N+1)*(log((double) N2)+1);
      
#if NR
      real_conv_fftnr(op2, op1, op1, N2);
#else
      real_conv_fftw(op2, op1, op1, N2);
#endif

      for(int n = 0; n < N+1; n++) op1[n] *= exp((theta[k-1]-theta[k])*n); 
      for(int n = N+1; n < N2; n++) op1[n] = 0;
      err *= MAX(1, exp((theta[k-1]-theta[k])*N));
    }

    delete[] op1; delete[] op2;

    return log(err*EPS) + theta[prob->get_K()-1]*N + total_log_sum;
  }
};

template<class prob_t> prob_t* theta2Fn<prob_t>::prob;  
template<class prob_t> int theta2Fn<prob_t>::N2;
template<class prob_t> int** theta2Fn<prob_t>::lat;
template<class prob_t> double theta2Fn<prob_t>::step;
template<class prob_t> double* theta2Fn<prob_t>::pu;

//
// Namespace for the variables and functions
// to use to compute theta in the sFFT and csFFT algorithm
//
class thetaFn {

 private:

  //
  // log_pmf = Log of single column lattice pmf
  // step = Step size for the lattice
  // s = LLR score for which theta is optimized
  // Q = Lattice size
  //  
  static double* log_pmf;
  static double step; static double s; static int Q;

 public:

  //
  // To initialize the namespace
  //  
  static void init(double* given_log_pmf, double given_step, double given_s, int given_Q) {

    log_pmf = given_log_pmf; step = given_step; s = given_s; Q = given_Q;    
  }

  //
  // Returns -theta*s + log(M(theta))
  //
  static double minimizationFn(double theta) {

    double log_mgf = LOGZERO;
    for(int It = 0; It < Q; It++)
      if(log_pmf[It] != LOGZERO)
	log_mgf = log_sum(log_mgf, log_pmf[It] + theta*step*It);

    return -theta*s + log_mgf;
  }      

  //
  // Returns sigma(theta)
  //
  static double get_sigma(double theta) {

    //
    // Let p[s] = exp(log_pmf[s/step]), then,
    // log_m0 = log(\sum_s p[s]exp(theta*s))
    // log_m1 = log(\sum_s s*p[s]exp(theta*s))
    // 
    double log_m0; double log_m1;
    log_m0 = log_m1 = LOGZERO;

    for(int i = 0; i < Q; i++) {

      log_m0 = log_sum(log_m0, log_pmf[i]+theta*step*i); 
      log_m1 = log_sum(log_m1, log_pmf[i]+theta*step*i+log(step*i));  
    }

    //
    // sigma = \sum_s (s-m1/m0)^2 p[s]exp(theta*s)/m0
    //
    double sigma = 0; double norm_m1 = exp(log_m1-log_m0);
    for(int i = 0; i < Q; i++) 
      sigma += pow(i*step-norm_m1, 2)*exp(log_pmf[i]+theta*i*step-log_m0);
    
    return sqrt(sigma);
  }

  static double get_s() { return s; }
};

inline lv_t coth(double x) {
  
  return (lv_t(x)+lv_t(-x))/(lv_t(x)-lv_t(-x));
}

inline lv_t csch(double x) {
  
  return lv_t(log(2))/(lv_t(x)-lv_t(-x));
}

inline lv_t cpsh(double x) {
  
  return (lv_t(x)+lv_t(-x))/lv_t(log(2));
}

class mw_thetaFn {

 private:
  
  static int m; static int n; 

 public:

  static double k;

  static void init(int given_m, int given_n, double given_k) 
    { m = given_m; n = given_n; k = given_k; }

  static double minimizationFn(double theta) {

    if(theta == 0)
      return 0;

    double log_mgf = 0;
    for(int i = 1; i <= m; i++)
      log_mgf += log(i/((double) n+i)) + 
	log_exp_minus_one(theta*(n+i)) - log_exp_minus_one(theta*i);

    return log_mgf - theta*k;
  }

  static double get_sigma(double theta, double* exp_theta_k) {
    
    if(theta == 0) return sqrt(m*n*(m+n+1)/12.0);
        
    double var = 0;   
    for(int x = 1; x <= m; x++) {
      
      double u = -(n+x)*exp_theta_k[n+x] + n*exp_theta_k[n+2*x] +
	x*exp_theta_k[x];
      double v = (1-exp_theta_k[n+x])*(1-exp_theta_k[x]);
      double du = -(n+x)*(n+x)*exp_theta_k[n+x] + n*(n+2*x)*exp_theta_k[n+2*x] +
	  x*x*exp_theta_k[x];
      double dv = (1-exp_theta_k[n+x])*(-x*exp_theta_k[x]) + 
	(-(n+x)*exp_theta_k[n+x])*(1-exp_theta_k[x]);
      var += (v*du-u*dv)/(v*v);
    }
    
    return sqrt(var);
  }

  static double get_log_sigma(double theta) {
    
    if(theta == 0) return 0.5*(log(m)+log(n)+log((m+n+1)/12));
        
    lv_t var(LOGZERO);   
    for(int j = 1; j <= m; j++) {
      lv_t ejt(j*theta); lv_t ejnt((j+n)*theta);
      lv_t j2(2*log(j)); lv_t jn2(2*log(j+n)); lv_t one(0);
      var = var + ejt.pow(2)*j2/(one-ejt).pow(2) + ejt*j2/(one-ejt) - 
	ejnt.pow(2)*jn2/(one-ejnt).pow(2) - ejnt*jn2/(one-ejnt);
 
      /*
      var = var + lv_t(j*theta)*
	((lv_t(0,-1)+lv_t(n*theta))*(lv_t(0,-1)+lv_t((2*j+n)*theta))*lv_t(2*log(j))
	 - lv_t(log(2)+n*theta)*(lv_t(0,-1)+lv_t(j*theta)).pow(2)*lv_t(log(j*n)) - 
	 lv_t(n*theta)*(lv_t(0,-1)+lv_t(j*theta)).pow(2)*lv_t(2*log(n))) /
	((lv_t(0,-1)+lv_t(j*theta)).pow(2)*(lv_t(0,-1)+lv_t((j+n)*theta)).pow(2));
      */
    }

    return 0.5*var.get_lv();
  }

  static double get_K3(double theta) {
    
    if(theta == 0) return LOGZERO;
        
    lv_t s(LOGZERO);   
    for(int x = 1; x <= m; x++)
      s = s + lv_t(3*log(x+n))*coth(theta*(x+n)/2)*csch(theta*(x+n)/2).pow(2) - 
	lv_t(3*log(x))*coth(theta*x/2)*csch(theta*x/2).pow(2);
    
    return s.get_sign()*exp(s.get_lv())/4;
  }

  static double get_K4(double theta) {
    
    if(theta == 0) return LOGZERO;
        
    lv_t s(LOGZERO);   
    for(int x = 1; x <= m; x++) 
      s = s + lv_t(4*log(x))*(lv_t(log(2))+cpsh(theta*x))*csch(theta*x/2).pow(4) -
	lv_t(4*log(x+n))*(lv_t(log(2))+cpsh(theta*(x+n)))*csch(theta*(x+n)/2).pow(4); 
    
    return s.get_sign()*exp(s.get_lv())/8;
  }
};

class mw_tied_thetaFn {

 private:
  
  static int m; static int n; 
  static int c; static int* size;
  static ee_t** choose; static double* log_mtheta; 

 public:

  static double k;

  static void init(int given_m, int given_n, double given_k,
		   int given_c, int* given_size, ee_t** given_choose) { 
    
    m = given_m; n = given_n; k = given_k;
    c = given_c; size = given_size; choose = given_choose;
  }

  static double minimizationFn(double theta) {

    if(theta == 0)
      return 0;

    int rank = 0;
    //ee_t* result = new ee_t[m+1];
    ee_t* result = new ee_t[m+n+1];
    for(int i = 0; i < MIN(size[0]+1, m+1); i++)
      result[i] = ee_t(theta*i*(2*rank+size[0]+1))*choose[size[0]][i];

    for(int i = 1; i < c; i++) {

      rank += size[i-1];
      ee_t* a = new ee_t[MIN(size[i]+1, m+1)];
      for(int j = 0; j < MIN(size[i]+1, m+1); j++)
	a[j] = ee_t(theta*j*(2*rank+size[i]+1))*choose[size[i]][j];

      //conv(a, result, result, MIN(size[i]+1, m+1), MIN(rank+size[i]+1, m+1));       
      conv(a, result, result, MIN(size[i]+1, m+1), rank+MIN(size[i]+1,m+1));
      //conv(a, result, result, size[i]+1, rank+MIN(size[i]+1,m+1));
    }

    //delete[](log_mtheta); log_mtheta = new double[m+1]; 
    delete[](log_mtheta); log_mtheta = new double[m+n+1]; 
    //for(int i = 0; i < m+1; i++)
    for(int i = 0; i < m+n+1; i++)    
      log_mtheta[i] = result[i].log_get();

    //cout << theta << " " << result[m].log_get()-choose[m+n][m].log_get() << " " 
    // << result[m].log_get()-theta*m*(m+1)-choose[m+n][m].log_get()-theta*k << endl;
    return result[m].log_get()-theta*m*(m+1)-choose[m+n][m].log_get()-theta*k;
    //return result[m].log_get()-choose[m+n][m].log_get()-theta*k;
  }

  static double theta2Fn(double theta) {

    double sum = LOGZERO; //cout << theta << ": ";
    //for(int i = 0; i < m+1; i++) {
      for(int i = 0; i < m+n+1; i++) {
      sum = log_sum(sum, log_mtheta[i]+theta*i);
      //cout << log_mtheta[i] + theta*i << " ";
    } //cout << endl;

    return sum - theta*(m);
  }

  static double get_sigma2(double theta) {

    double m0 = LOGZERO;
    double m1 = LOGZERO;
    double m2 = LOGZERO;

    //for(int i = 0; i < m+1; i++) {
    for(int i = 0; i < m+n+1; i++) {
      m0 = log_sum(m0, log_mtheta[i]+theta*i);
      if(i > 0)
	m1 = log_sum(m1, log_mtheta[i]+theta*i+log(i));
      if(i > 0) 
	m2 = log_sum(m2, log_mtheta[i]+theta*i+2*log(i));
      //cout << i << "," << log_mtheta[i] << "," << log_mtheta[i] + theta*i << " ";
    } //cout << endl;

    return sqrt(exp(m2-m0)-exp(2*(m1-m0)));
  }

  static double get_sigma(double theta) {
    
    ee_t** p0 = new ee_t*[c]; ee_t** p1 = new ee_t*[c]; ee_t** p2 = new ee_t*[c];
      
    int rank = 0;
    for(int i = 0; i < c; i++) {
      
      p0[i] = new ee_t[m+1]; p1[i] = new ee_t[m+1]; p2[i] = new ee_t[m+1];
      for(int j = 0; j < MIN(size[i]+1, m+1); j++) {
	double s = j*(2*rank+size[i]+1);
	p0[i][j].log_set(theta*s + choose[size[i]][j].log_get()); 
      	p1[i][j].log_set(theta*s + choose[size[i]][j].log_get() + log(s));
	p2[i][j].log_set(theta*s + choose[size[i]][j].log_get() + 2*log(s)); 
      }

      rank += size[i];
    }

    ee_t *out1 = new ee_t[m+1]; ee_t *out2 = new ee_t[m+1]; ee_t two(log(2.0));

    //
    // m0 = M(theta)
    // m1 = M'(theta)
    // m2 = M''(theta)
    //
    ee_t* m0 = p0[0]; ee_t* m1 = p1[0]; ee_t* m2 = p2[0];
    for(int i = 1; i < c; i++) {

      conv(p0[i], m2, m2, size[i]+1, m+1);
      conv(p1[i], m1, out1, size[i]+1, m+1);
      conv(p2[i], m0, out2, size[i]+1, m+1);

      for(int n = 0; n < m+1; n++)
	m2[n] += out2[n] + two*out1[n];

      conv(p0[i], m1, m1, size[i]+1, m+1);
      conv(p1[i], m0, out1, size[i]+1, m+1);
 
      for(int n = 0; n < m+1; n++) 
	m1[n] += out1[n];

      conv(p0[i], m0, m0, size[i]+1, m+1);
    }

    //
    // m1N = Normalized first moment
    // m2N = Normalized second moment
    //
    double m1N = exp(m1[m].log_get()-m0[m].log_get());
    double m2N = exp(m2[m].log_get()-m0[m].log_get());

    //for(int i = 0; i < c-1; i++) {
    //  delete[](p0[i]); delete[](p1[i]); delete[](p2[i]);
    //}

    cout << "M's : " << m1N << " " << m2N << endl;

    // clean up
    delete [] out1; 
    delete [] out2;
    return (m1N*m1N >= m2N ? EPS : sqrt(m2N-m1N*m1N));
  }

  static void get_K(double theta, double& K3, double& K4) {
    
    ee_t** p0 = new ee_t*[c]; ee_t** p1 = new ee_t*[c]; 
    ee_t** p2 = new ee_t*[c]; ee_t** p3 = new ee_t*[c];
    ee_t** p4 = new ee_t*[c];
      
    int rank = 0;
    for(int i = 0; i < c; i++) {
      
      p0[i] = new ee_t[m+1]; p1[i] = new ee_t[m+1]; 
      p2[i] = new ee_t[m+1]; p3[i] = new ee_t[m+1];
      p4[i] = new ee_t[m+1];

      for(int j = 0; j < MIN(size[i]+1, m+1); j++) {
	double s = j*(2*rank+size[i]+1);
	p0[i][j].log_set(theta*s + choose[size[i]][j].log_get()); 
      	p1[i][j].log_set(theta*s + choose[size[i]][j].log_get() + log(s));
	p2[i][j].log_set(theta*s + choose[size[i]][j].log_get() + 2*log(s)); 
	p3[i][j].log_set(theta*s + choose[size[i]][j].log_get() + 3*log(s)); 
	p4[i][j].log_set(theta*s + choose[size[i]][j].log_get() + 4*log(s)); 
      }

      rank += size[i];
    }

    ee_t *out1 = new ee_t[m+1]; 
    ee_t *out2 = new ee_t[m+1]; 
    ee_t *out3 = new ee_t[m+1]; 
    ee_t *out4 = new ee_t[m+1]; 
    ee_t two(log(2.0)); ee_t three(log(3.0)); ee_t four(log(4.0)); ee_t six(log(6.0));

    //
    // m0 = M(theta)
    // m1 = M'(theta)
    // m2 = M''(theta)
    //
    ee_t* m0 = p0[0]; ee_t* m1 = p1[0]; ee_t* m2 = p2[0]; ee_t* m3 = p3[0]; ee_t* m4 = p4[0];
    for(int i = 1; i < c; i++) {

      conv(p0[i], m4, m4, size[i]+1, m+1);
      conv(p1[i], m3, out1, size[i]+1, m+1);
      conv(p2[i], m2, out2, size[i]+1, m+1);
      conv(p3[i], m1, out3, size[i]+1, m+1);
      conv(p4[i], m0, out4, size[i]+1, m+1);
      for(int n = 0; n < m+1; n++)
	m4[n] += out4[n] + four*out3[n] + six*out2[n] + four*out1[n];

      conv(p0[i], m3, m3, size[i]+1, m+1);
      conv(p1[i], m2, out1, size[i]+1, m+1); 
      conv(p2[i], m1, out2, size[i]+1, m+1); 
      conv(p3[i], m0, out3, size[i]+1, m+1); 
      for(int n = 0; n < m+1; n++) 
	m3[n] += out3[n] + three*out2[n] + three*out1[n];

      conv(p0[i], m2, m2, size[i]+1, m+1);
      conv(p1[i], m1, out1, size[i]+1, m+1);
      conv(p2[i], m0, out2, size[i]+1, m+1);
      for(int n = 0; n < m+1; n++)
	m2[n] += out2[n] + two*out1[n];

      conv(p0[i], m1, m1, size[i]+1, m+1);
      conv(p1[i], m0, out1, size[i]+1, m+1); 
      for(int n = 0; n < m+1; n++) 
	m1[n] += out1[n];

      conv(p0[i], m0, m0, size[i]+1, m+1);
    }

    //
    // m1N = Normalized first moment
    // m2N = Normalized second moment
    //
    double m1N = exp(m1[m].log_get()-m0[m].log_get());
    double m2N = exp(m2[m].log_get()-m0[m].log_get());
    double m3N = exp(m3[m].log_get()-m0[m].log_get());
    double m4N = exp(m4[m].log_get()-m0[m].log_get());

    //for(int i = 0; i < c-1; i++) {
    //  delete[](p0[i]); delete[](p1[i]); delete[](p2[i]);
    //}

    cout << "M's : " << m1N << " " << m2N << " " << m3N << " " << m4N << endl;
    K3 = 2*pow(m1N, 3) - 3*m2N*m1N + m3N;
    K4 = -6*pow(m1N, 4) + 12*pow(m1N, 2)*m2N - 3*pow(m2N, 2) - 4*m1N*m3N + m4N;

    // clean up
    delete [] out1; 
    delete [] out2;
    delete [] out3; 
    delete [] out4;
  }
};

#endif
