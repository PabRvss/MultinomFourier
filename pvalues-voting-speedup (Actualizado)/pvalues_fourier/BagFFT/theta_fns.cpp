#pragma once
#include "theta_fns.h"

double* thetaFn::log_pmf;
double thetaFn::step; double thetaFn::s; 
int thetaFn::Q;

int mw_thetaFn::m; int mw_thetaFn::n; double mw_thetaFn::k;
int mw_tied_thetaFn::m; int mw_tied_thetaFn::n; double mw_tied_thetaFn::k;
int mw_tied_thetaFn::c; int* mw_tied_thetaFn::size; ee_t** mw_tied_thetaFn::choose;
double* mw_tied_thetaFn::log_mtheta;
