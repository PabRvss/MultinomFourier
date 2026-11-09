#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

#include <math.h>
#include "complex.h"
#include "fft.h"

//
// Computes the convolution of a[0..(n-1)] and b[0..(to_fill-1)]
// and writes the result in result[0..(to_fill-1)].
// The computation is done "naively" i.e. without using FFT's.
// Note that b and result can be the same vector.
//
template<class T>
void conv(T* a, T* b, T* result, int n, int to_fill) {

  for(int i = to_fill-1; i >= 0; i--) {
    
    result[i] = a[0]*b[i];
    
    for(int j = 1; j <= MIN(i, n-1); j++)              
      result[i] += a[j]*b[i-j];
  }
}

// Returns the self-convolution of a[0..(n-1)] 
// to the given power.
// The computation is done "naively".
//
template<class T>
T* self_conv(T* a, int n, int power) {
 
  T* result = new T[n*power];
  for(int i = 0; i < n; i++) 
    result[i] = a[i];
  
  for(int p = 2; p <= power; p++)
    conv<T>(a, result, result, n, n*p);
	 
  return result;
}
 

//
// Computes the fft of a[0..(n-1)].
// Based on Numerical recipes code.
//
inline void fftnr(complex_t* a, int n, int sign) {

  fft(((double*) a)-1, n, sign);
}

//
// Sets result to the convolution of a and b where
// they are all vectors of size n and n is large
// enough to hold the result (also a power of 2).
// Computed using FFTs and based on Numerical recipes code.
//
inline void conv_fftnr(complex_t* a , complex_t* b, complex_t* result, int n) {

  fft(((double*) a)-1, n, -1); fft(((double*) b)-1, n, -1);
      
  for(int x = 0; x < n; x++)
    result[x] = a[x]*b[x]/((double) n);	

  fft(((double*) result)-1, n, 1);
}

//
// Sets result to the convolution of a and b where
// they are all real vectors of size n and n is large
// enough to hold the result.
// Computed using FFTs and based on Numerical recipes code.
//
inline void real_conv_fftnr(double* a , double* b, double* result, int n) {

  fft_real_conv(a-1, n/2, b-1, n/2, 1, result-1);
}

//
// FFTW code. To be compiled if NR = 0.
//

#if NR == 0

#include <fcntl.h>
#include <unistd.h>
#include "fftw3.h"

//
// Returns an appropriate structure for
// read and write locks to pass to the function fcntl.
//
inline struct flock* file_lock(short type, short whence) {

  static struct flock ret;
  ret.l_type = type; ret.l_start = 0; ret.l_whence = whence;
  ret.l_len = 0; ret.l_pid = getpid();
  return &ret;
}

//
// For importing the wisdom file for FFTW
// before computing the first fft.
//
static int wisdom_init = 0;
inline void init_fftw() {

  if(!wisdom_init) {
    FILE* wisdom = fopen("wisdom", "r");
    if (wisdom) {
      int w = open("wisdom", O_RDONLY);
      if (w >= 0) {
          fcntl(w, F_SETLKW, file_lock(F_RDLCK, SEEK_SET));
          fftw_import_wisdom_from_file(wisdom);
          fcntl(w, F_SETLKW, file_lock(F_RDLCK, SEEK_SET));
          close(w);
        }
      fclose(wisdom);
    }
    wisdom_init = 1;
  }
}

//
// To update the wisdom file with newly acquired wisdom.
//
inline void save_fftw() {

    FILE* wisdom = fopen("wisdom", "w");
    if (!wisdom) return;
    int w = open("wisdom", O_WRONLY);
    if (w >= 0) {
      fcntl(w, F_SETLKW, file_lock(F_WRLCK, SEEK_SET));
      fftw_export_wisdom_to_file(wisdom);
      fcntl(w, F_SETLKW, file_lock(F_WRLCK, SEEK_SET));
      close(w);
    }
}

//
// Computes the fft of a[0..(n-1)].
// Based on fftw code.
//
inline void fftw(complex_t* a, int n, int sign) {

  init_fftw();
  fftw_complex* in = new fftw_complex[n]; fftw_complex* out = new fftw_complex[n];
  fftw_plan fft = fftw_plan_dft_1d(n, in, out, sign, FFTW_MEASURE);
  save_fftw();

  for(int i = 0; i < n; i++) {
    in[i][0] = a[i].real();
    in[i][1] = a[i].imag();
  }
  fftw_execute(fft);

  for(int i = 0; i < n; i++)
    a[i].set(out[i][0], out[i][1]);

  fftw_destroy_plan(fft);
  delete[] in; delete[] out;
}

//
// Sets result to the convolution of a and b where
// they are all vectors of size n and n is large
// enough to hold the result.
// Computed using FFTs and based on fftw code.
//
inline void conv_fftw(complex_t* a , complex_t* b, complex_t* result, int n) {

  static int conv_init = 0; static int cur_n = 0;
  static fftw_complex* in1 = 0; static fftw_complex* in2 = 0;
  static fftw_complex* out1 = 0; static fftw_complex* out2 = 0;
  static fftw_plan fft1 = 0; static fftw_plan fft2 = 0; 
  static fftw_plan ifft = 0;

  //
  // Avoids creating new arrays and plans everytime
  // 
  if(!conv_init || n != cur_n) {

    init_fftw();
    delete[](in1); delete[](in2); delete[](out1); delete[](out2);
    fftw_destroy_plan(fft1); fftw_destroy_plan(fft2);
    fftw_destroy_plan(ifft);
    in1 = new fftw_complex[n]; in2 = new fftw_complex[n];
    out1 = new fftw_complex[n]; out2 = new fftw_complex[n];
    fft1 = fftw_plan_dft_1d(n, in1, out1, FFTW_FORWARD, FFTW_MEASURE);
    fft2 = fftw_plan_dft_1d(n, in2, out2, FFTW_FORWARD, FFTW_MEASURE);
    ifft = fftw_plan_dft_1d(n, in1, out1, FFTW_BACKWARD, FFTW_MEASURE);
    conv_init = 1; cur_n = n;
    save_fftw();
  }

  for(int i = 0; i < n; i++) {
    in1[i][0] = a[i].real();
    in1[i][1] = a[i].imag();
  }
  fftw_execute(fft1);

  for(int i = 0; i < n; i++) {
    in2[i][0] = b[i].real();
    in2[i][1] = b[i].imag();
  }
  fftw_execute(fft2);
  
  for(int i = 0; i < n; i++) {
    in1[i][0] = (out1[i][0]*out2[i][0]-out1[i][1]*out2[i][1])/n;
    in1[i][1] = (out1[i][1]*out2[i][0]+out1[i][0]*out2[i][1])/n;
  }  
  fftw_execute(ifft);

  for(int i = 0; i < n; i++)
    result[i].set(out1[i][0], out1[i][1]);
}

//
// Sets result to the convolution of a and b where
// they are all real vectors of size n and n is large
// enough to hold the result.
// Computed using FFTs and based on fftw code.
//
inline void real_conv_fftw(double* a , double* b, double* result, int n) {

  static int conv_init = 0; static int cur_n = 0;
  static double* in1 = 0; static double* in2 = 0;
  static fftw_complex* out1 = 0; static fftw_complex* out2 = 0;
  static fftw_plan fft1 = 0; static fftw_plan fft2 = 0; 
  static fftw_plan ifft = 0;

  //
  // Avoids creating new arrays and plans everytime
  // 
  if(!conv_init || n != cur_n) {

    init_fftw();
    delete[](in1); delete[](in2); delete[](out1); delete[](out2);
    fftw_destroy_plan(fft1); fftw_destroy_plan(fft2);
    fftw_destroy_plan(ifft);
    in1 = new double[n]; in2 = new double[n];
    out1 = new fftw_complex[n]; out2 = new fftw_complex[n];
    fft1 = fftw_plan_dft_r2c_1d(n, in1, out1, FFTW_MEASURE);
    fft2 = fftw_plan_dft_r2c_1d(n, in2, out2, FFTW_MEASURE);
    ifft = fftw_plan_dft_c2r_1d(n, out1, in1, FFTW_MEASURE);
    conv_init = 1; cur_n = n;
    save_fftw();
  }

  for(int i = 0; i < n; i++)
    in1[i] = a[i];
  fftw_execute(fft1);

  for(int i = 0; i < n; i++)
    in2[i] = b[i];
  fftw_execute(fft2);
  
  for(int i = 0; i < n; i++) {
    double re = (out1[i][0]*out2[i][0]-out1[i][1]*out2[i][1])/n;
    double im = (out1[i][1]*out2[i][0]+out1[i][0]*out2[i][1])/n;
    out1[i][0] = re; out1[i][1] = im;
  }  
  fftw_execute(ifft);

  for(int i = 0; i < n; i++)
    result[i] = in1[i];
}

//
// Returns the cyclic self-convolution of a[0..(n-1)] 
// to the given power with the given period.
// Computed using FFTs and based on fftw code.
//
double* self_conv_fftw(double* a, int n, int power, int period);

//
// Returns the cyclic self-convolution of a[0..(period-1)] 
// to the all the powers between min_power and max_power, with the given period.
// Computed using FFTs and based on fftw code.
//
double** multi_self_conv_fftw(double* a, int min_power, int max_power, int period);

#endif

#endif
