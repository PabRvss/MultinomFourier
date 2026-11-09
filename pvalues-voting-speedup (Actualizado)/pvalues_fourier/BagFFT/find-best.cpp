#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

inline int charToNumber(char a) {

  switch(a) {
  case 'A': return 0;
  case 'C': return 1;
  case 'G': return 2;
  case 'T': return 3;
  default: cout << "Error: Non ACGT character" << endl; return 0;
  }
}

inline char numberToChar(int i) {

  static char characters[] = {'A', 'C', 'G', 'T'}; 
  return characters[i];
}

void read_sequences(char* filename, vector<vector<int> >& sequences, double* pu) {

  ifstream sequence_file(filename);

  string in; 
  sequence_file >> in;  

  do {
    string sequence;
    while((sequence_file >> in) && in[0] != '>')
      sequence += in;
 
    vector<int> numeric_representation;
    for(string::iterator it = sequence.begin(); it != sequence.end(); it++) {
      numeric_representation.push_back(charToNumber(*it));
      pu[charToNumber(*it)]++;
    }
    sequences.push_back(numeric_representation);
  
  } while(sequence_file);
}

const int K = 4; 
int L; int N;

vector<vector<int> > sequences;
vector<int> best_motif; double best_score;
vector<int> curr_motif; 
vector<vector<int> > curr_profile;

double** score_matrix;
double** upd_score_matrix;

inline double update_profile(vector<int>::iterator it) {

  double upd_score = 0; 
  vector<vector<int> >::iterator col = curr_profile.begin();
  for(int i = 0; i < L; i++, it++, col++)
    upd_score += upd_score_matrix[*it][(*col)[*it]++];

  return upd_score;
}

inline void reset_profile(vector<int>::iterator it) {

  vector<vector<int> >::iterator col = curr_profile.begin();
  for(int i = 0; i < L; i++, it++, col++)
    (*col)[*it]--;
}
      
inline double best_extended_score(vector<int>::iterator it, int left) {

  double ext_score = 0; 
  vector<vector<int> >::iterator col = curr_profile.begin();
  for(int i = 0; i < L; i++, it++, col++) {
    //int max_k = 0; 
    //int max_count = (*col)[0];
    double max_upd = score_matrix[0][(*col)[0]+left] -
      score_matrix[0][(*col)[0]];
    for(int k = 1; k < K; k++)
      /*
      if((*col)[k] > max_count) {
	max_count = (*col)[k];
	max_k = k;
	}*/
      max_upd = max(max_upd, score_matrix[k][(*col)[k]+left] -
		    score_matrix[k][(*col)[k]]);
    
    //ext_score += score_matrix[max_k][max_count+left] -
    //score_matrix[max_k][max_count];
    ext_score += max_upd;
  }

  return ext_score;
}

void find_best_motif(int seq_no, double curr_score) {
//void find_best_motif(int seq_no, double curr_score, string pos) {

  if(seq_no == N) {

    //cout << pos << endl;
    if(curr_score > best_score) {
      best_score = curr_score;
      best_motif = curr_motif;
      cout << best_score << endl;
    }
  }
  else {

    //cout << "At seq: " << seq_no << endl;
    
    int i = 0;
    for(vector<int>::iterator it = sequences[seq_no].begin();
      it != sequences[seq_no].end()-L+1; it++, i++) {
      
      //if(seq_no == 0)
      //cout << seq_no << "," << i << " ";
      
      double upd_score = update_profile(it);
      if(best_extended_score(it, N-seq_no-1)+curr_score+upd_score 
	 > best_score) {
	curr_motif.push_back(i);
	find_best_motif(seq_no+1, curr_score + upd_score);
	//find_best_motif(seq_no+1, curr_score + upd_score, pos + char('A' + seq_no) + ' ');
      curr_motif.pop_back();
      }
      //else
      //cout << pos << endl;
      
      reset_profile(it);
    }
  }
}
    
int main(int argc, char** argv) {

  double pu[K];
  for(int i = 0; i < K; i++)
    pu[i] = 0;

  L = atoi(argv[2]);
  curr_profile.resize(L, vector<int>(K, 0));

  read_sequences(argv[1], sequences, pu);
  N = sequences.size();

  best_score = atof(argv[3]);
  
  double sum = 0;
  for(int i = 0; i < K; i++)
    sum += pu[i];

  for(int i = 0; i < K; i++)
    pu[i] /= sum;

  score_matrix = new double*[K]; upd_score_matrix = new double*[K];
  for(int i = 0; i < K; i++) {

    score_matrix[i] = new double[N+1];
    upd_score_matrix[i] = new double[N];
    score_matrix[i][0] = 0;
    for(int j = 1; j <= N; j++) { 
      score_matrix[i][j] = j*log(j/pu[i]/N);
      upd_score_matrix[i][j-1] = score_matrix[i][j]-score_matrix[i][j-1];
    }
  }
  
  //cout << N << endl;

  find_best_motif(0, 0);
  //find_best_motif(0, 0, "");

  if(!best_motif.empty()) {
    for(int i = 0; i < N; i++)
      cout << best_motif[i] << " ";
    cout << endl << endl;
    
    for(int i = 0; i < N; i++) {
      for(int j = 0; j < L; j++)
	cout << numberToChar(sequences[i][best_motif[i]+j]);
      cout << endl;
    } cout << endl;

    cout << "Score: " << best_score << endl; 
  }
  else
    cout << "The min bound was too high!" << endl;

  return 0;
}

