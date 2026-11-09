#ifndef _LOG_VALUE_H_
#define _LOG_VALUE_H_

#include "utils.h"
#include <math.h>
#include <iostream>

using namespace std;

class lv_t {

 private:

  double lv;
  int sign;

 public:
  
  inline lv_t() { lv = LOGZERO; sign = 1; }
  inline lv_t(double x) { lv = x; sign = 1; }
  inline lv_t(double x, int i) { lv = x; sign = i; }

  inline double get_lv() { return lv; }
  inline int get_sign() { return sign; }
  inline double log_get() { return lv; }

  inline lv_t& operator=(double a) 
    { lv = log(fabs(a)); sign = (int) (a/fabs(a)); return *this; }

  inline lv_t& operator=(const lv_t& a) 
    { lv = a.lv; sign = a.sign; return *this; }

  inline lv_t operator*(const lv_t& a) 
    { return lv_t(lv+a.lv, sign*a.sign); }

  inline lv_t operator/(const lv_t& a) 
    { return lv_t(lv-a.lv, sign*a.sign); }

  inline lv_t operator+(const lv_t& a) {

    if(sign == a.sign)
      return lv_t(log_sum(lv, a.lv), sign);
    else if(lv > a.lv)
      return lv_t(lv + log(1-exp(a.lv-lv)), sign);
    else if(a.lv > 2*EPS + lv) 
      return lv_t(a.lv + log(1-exp(lv-a.lv)), a.sign);
    else
      return lv_t(LOGZERO, 1);
  }

  inline lv_t operator-(const lv_t& a) 
    { return operator+(lv_t(a.lv, -a.sign)); }

  inline lv_t& operator-=(const lv_t& a)
    { return *this = *this - a; }

  inline lv_t& operator+=(const lv_t& a)
    { return *this = *this + a; }

  inline lv_t pow(int x) { return lv_t(lv*x, (x%2 == 0 ? 1 : sign)); }
    
};

inline double log(lv_t& a) { return a.log_get(); }

#endif
