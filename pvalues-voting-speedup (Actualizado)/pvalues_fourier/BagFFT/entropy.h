#ifndef _ENTROPY_H_
#define _ENTROPY_H_

#define NOT_COMPUTED 0

//
// Function: Computes the lattice pmf (log-values in an array of size
//           L*Q) for a sum of scores based on the NC algorithm.
//
// Parameters: log_pmf = Log of lattice pmf for single sample
//             L = Number of i.i.d. samples
//             Q = Lattice size for single sample
//
double* NC(double* log_pmf, int Q, int L);

//
// Function: Computes the p-value for a sum of scores based on the 
//           NC algorithm with pruning and lists.
//
// Parameters: log_pmf = Log of lattice pmf for single sample
//             L = Number of i.i.d. samples
//             Q = Lattice size for single sample
//             s = Score to compute p-value for
//
double* NCpruned(double* log_pmf, int Q, int L, double s);

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
double* sFFT(double* log_pmf, int L, double s, int Q, double step, double& theta, double& log_mgf);

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
double* csFFT(double* log_pmf, int L, double s, int Q, double step, double& theta, double& log_mgf);

#endif
