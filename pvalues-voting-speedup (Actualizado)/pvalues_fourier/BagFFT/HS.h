#ifndef _HS_H_
#define _HS_H_

//
// Function: Computes and returns the entire lattice pmf (log-values in an array of
//           size Q) for the multinomial llr score based on the shifted-hirji algorithm.
//
// Parameters: N = Number of objects 
//             K = Number of bins 
//             pu = Null multinomial distribution 
//             Q = Lattice size 
//             step = Step size for the lattice (set by the function)
//
double* HSarray(int N, int K, double *pu, int Q, double& step);

//
// Function: Computes and returns the entire lattice pmf (log-values in an array of
//           size Q) for the multinomial llr score based on the shifted-hirji algorithm
//           using lists.
//
// Parameters: N = Number of objects 
//             K = Number of bins 
//             pu = Null multinomial distribution 
//             Q = Lattice size 
//             step = Step size for the lattice (set by the function)
//             index = Array of indices for non-zero pmf entries (set by the function)
//             size = Size of the index array (set by the function)
//
double* HSlist(int N, int K, double *pu, int Q, double& step, int*& index, int& size);

//
// Function: Computes and returns the p-value for the multinomial llr score s 
//           based on the shifted-hirji algorithm with lists and pruning.
//
// Parameters: N = Number of objects 
//             K = Number of bins 
//             s = LLR score
//             pu = Null multinomial distribution 
//             Q = Lattice size 
//             log_min_pval = log lower bound for p-value (set by the function)
//             log_max_pval = log upper bound for p-value (set by the function)
//
void HSsingle(int N, int K, double s, double *pu, int Q, double& log_min_pval, double& log_max_pval);

//
// Function: Computes and returns the entire lattice pmf for all values of N 
// (N arrays of log-values in arrays of size Q) for the multinomial llr score 
// based on the shifted-hirji algorithm.
//
// Parameters: N = Number of objects 
//             K = Number of bins 
//             pu = Null multinomial distribution 
//             Q = Lattice size 
//             step = Step size for the lattice (set by the function)
//
double** HSmultiarray(int N, int K, double *pu, int Q, double& step);

#endif 
