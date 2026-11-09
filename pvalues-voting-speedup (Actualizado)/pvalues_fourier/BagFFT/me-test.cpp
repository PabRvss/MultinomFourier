#include "utils.h"
#include "motif_evaluator.h"

int main(int argc, char** argv) {

  //
  // Reading input parameters from the command line
  //
  int N; int K; int L; double s; int Q; double* pu;
  read_entropy_parameters(argc, argv, N, K, L, s, Q, pu);

  motif_evaluator m;
  m.init(N, K, L, L, pu, Q);

  //for(int i = 2; i <= N; i++)
  //  m.setN(i);
  m.setN(N);

  cout << m.get_pvalue_careful(0, s) << endl;
}
