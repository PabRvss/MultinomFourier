#ifndef _complex_t_H_
#define _complex_t_H_

class complex_t;

#include <math.h>
#include "utils.h"

//
// Class for representing complex values
//
class complex_t {

 private:

  double re;
  double im;

 public:

  inline void set(double given_re, double given_im) { re = given_re; im = given_im; }  

  //
  // Constructors
  //
  inline complex_t() { set(0, 0); }
  inline complex_t(double given_re, double given_im) { set(given_re, given_im); }  
  inline complex_t(const complex_t& c) { set(c.re, c.im); }

  inline complex_t& operator=(const complex_t& c) { set(c.re, c.im); return *this; }

  inline double real() { return re; }
  inline double imag() { return im; }

  //
  // Returns the conjugate of the number
  //
  inline complex_t conj() const { return complex_t(re, -im); }

  inline complex_t operator-(const complex_t& c) const { return complex_t(re-c.re, im-c.im); }
  inline complex_t operator-(double d) const { return complex_t(re-d, im); }
  inline complex_t operator+(const complex_t& c) const { return complex_t(re+c.re, im+c.im); }
  inline complex_t& operator+=(const complex_t& c) { set(re+c.re, im+c.im); return *this; }

  inline complex_t operator*(double given_re) const { return complex_t(re*given_re, im*given_re); }
  inline complex_t operator*(const complex_t& c) const { return complex_t(re*c.re - im*c.im, re*c.im + im*c.re); }
  inline complex_t& operator*=(const complex_t& c) { set(re*c.re - im*c.im, re*c.im + im*c.re); return *this; }

  inline complex_t operator/(const complex_t& c) const { 
    return complex_t((re*c.re+im*c.im)/(c.re*c.re+c.im*c.im), 
		   (im*c.re-re*c.im)/(c.re*c.re+c.im*c.im)); 
  }

  inline complex_t operator/(double given_re) const { return complex_t(re/given_re, im/given_re); }

  //
  // Returns (re + im*i)*exp(log_shift) 
  // (computes it carefully to preserve accuracy)
  //
  inline complex_t shift(double log_shift) 
    { return complex_t(sign(re)*exp(log(fabs(re))+log_shift), sign(im)*exp(log(fabs(im))+log_shift)); }
};

inline double abs(complex_t a) { return sqrt(a.real()*a.real()+a.imag()*a.imag()); }
inline complex_t exp(complex_t a) { return complex_t(cos(a.imag()), sin(a.imag()))*exp(a.real()); }
inline complex_t sinh(complex_t a) { return complex_t(sinh(a.real())*cos(a.imag()), cosh(a.real())*sin(a.imag())); }

#endif
