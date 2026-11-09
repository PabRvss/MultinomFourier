#include "utils.h"
#include "motif_evaluator.h"

#include <string>
#include <fstream>
#include <list>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <algorithm>

#define PR1 0

//
// CONS = 1 should be used with FIXED = 1
//
#define CONS 0

#define BIG_EVALUE 1e100

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
  
void read_sequences(char* filename, vector<vector<int> >& sequences, vector<string>& names, double* pu) {

  ifstream sequence_file(filename);

  string in; 
  sequence_file >> in; 

  do {

    names.push_back(in); 

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

typedef pair<int, int> subsequence;

class motif {

public:

  map<int, int> starting_positions;
  double min_score; double max_score;

#if FIXED
  double log_alignment_count;
#else
  vector<double> log_alignment_count;
#endif

  double log_evalue;

  static int** profile;
  static deque<double> scores;

public:
  void update_evalue() {

    double score = min_score;     
    op_pos = 0; op_score = lower_bounds[0];
    log_evalue = (score > lower_bounds[0] ? op_score = score, evaluator.get_pvalue(0, score) : BIG_EVALUE);

#if !FIXED
    log_evalue += log_alignment_count[0];
#endif

    int Lindex = 1;
    for(int i = minL; i < maxL; i++, Lindex++) {
      
      score += scores[i];

      if(score > lower_bounds[Lindex]) {

#if FIXED
	double curr_log_evalue = evaluator.get_pvalue(Lindex, score);
#else
	double curr_log_evalue = evaluator.get_pvalue(Lindex, score) + log_alignment_count[Lindex];
#endif

	if(log_evalue > curr_log_evalue) {
	  log_evalue = curr_log_evalue;
	  op_score = score; op_pos = Lindex;
	}
      }
    }

#if FIXED
    log_evalue += log_alignment_count;
#endif
  }

private:
  void simple_evalue() {
#if CONS
    log_evalue = -max_score;
#else
    
#if FIXED
    log_evalue = evaluator.get_pvalue(max_minus_minL, max_score) + log_alignment_count;
#else
    log_evalue = evaluator.get_pvalue(max_minus_minL, max_score) + log_alignment_count[max_minus_minL];
#endif

#endif
  }

  void add_subsequence(const subsequence& s) {

    starting_positions.insert(s);
#if FIXED
    log_alignment_count += log_lengths[s.first];
#else
    for(int L = 0; L <= max_minus_minL; L++)
      log_alignment_count[L] += log_lengths[s.first][L];
#endif
  }

public:

  static motif_evaluator evaluator;
  static int K; static int minL; static int maxL; static int max_minus_minL;
  
  static vector<vector<int> >* sequences;

#if FIXED
  static double* log_lengths; 
#else
  static double** log_lengths; 
#endif

  static double op_score; static int op_pos;
  static vector<double> lower_bounds;

  static double nlogn; 

  motif(const subsequence& s1, const subsequence& s2) 
#if FIXED
    : log_alignment_count(0)
#endif
  {
#if !FIXED
    log_alignment_count.resize(max_minus_minL+1);
    for(int i = 0; i <= max_minus_minL; i++)
      log_alignment_count[i] = 0;
#endif
    add_subsequence(s1); add_subsequence(s2);
    set_profile(); 

#if FIXED
    simple_evalue();
#else
    update_evalue();
#endif    
  }

  motif(const motif& m, int seq1, int seq2, int k) 
    : starting_positions(m.starting_positions), log_evalue(m.log_evalue), 
      log_alignment_count(m.log_alignment_count)
  { starting_positions[seq1] += k; starting_positions[seq2] += k; }

  motif(const motif& m, const subsequence& s, double new_log_evalue) 
    : starting_positions(m.starting_positions), log_evalue(new_log_evalue), 
      log_alignment_count(m.log_alignment_count) 
  { add_subsequence(s); } 

  void increment(int seq1, int seq2, int c1, int c2) {

    //scores.push_back(evaluator.pairUpd[c1][c2]);  
    scores.push_back((c1 != c2 ? evaluator.Iupd[c1][0] + evaluator.Iupd[c2][0] : evaluator.Iupd[c1][0]+evaluator.Iupd[c1][1]) + nlogn);  
    min_score -= scores.front(); min_score += scores[minL];
    //max_score -= scores.front(); 
    scores.pop_front();
    //max_score += scores.back();

#if FIXED
    simple_evalue();
#else
    update_evalue();
#endif    

    //min_score -= scores.front(); scores.pop_front();
    //if(scores.size() > minL)
    //scores[minL] = evaluator.pairUpd[c1][c2];
    //else
    //scores.push_back(evaluator.pairUpd[c1][c2]);  
    //min_score += scores[minL];
    //simple_evalue();
  }

  double get_log_evalue() const 
    { return log_evalue; }

  double get_max_score() const 
    { return max_score; }

#if FIXED
  double get_log_alignment_count() const 
    { return log_alignment_count; }
#else
  double get_log_alignment_count(int L_minus_min) const 
    { return log_alignment_count[L_minus_min]; }
#endif

  bool operator<(const motif& m) const { return get_log_evalue() < m.get_log_evalue(); }
  bool operator<(double given_log_evalue ) const { return get_log_evalue() > given_log_evalue; }
  
  bool operator==(motif& m) { 

    if(get_log_evalue() != m.get_log_evalue())
      return false; 
    else
      for(map<int, int>::iterator it1 = starting_positions.begin(),
	    it2 = m.starting_positions.begin();
	  it1 != starting_positions.end(); it1++, it2++)
	if(it1->first != it2->first || it1->second != it2->second)
	  return false;

    return true;
  }

  map<int, int>& get_starting_positions() { return starting_positions; }

  void set_profile() {

     if(profile == 0) {
      
      profile = new int*[maxL];

      for(int i = 0; i < maxL; i++) 	
	profile[i] = new int[K];

      scores.resize(maxL);
    }

    for(int i = 0; i < maxL; i++)
      for(int j = 0; j < K; j++)
	profile[i][j] = 0;

    for(int i = 0; i < maxL; i++) 	
      scores[i] = nlogn;

    for(map<int, int>::iterator it1 = starting_positions.begin();
	it1 != starting_positions.end(); it1++) {
      
      vector<int>::iterator it2 = (*sequences)[it1->first].begin()+it1->second;
      for(int i = 0; i < maxL; i++, it2++)    
	scores[i] += evaluator.Iupd[*it2][profile[i][*it2]++];
    }

    min_score = 0;
    for(int i = 0; i < minL; i++)    
      min_score += scores[i];

    max_score = min_score;
    for(int i = minL; i < maxL; i++)
      max_score += scores[i];
  }
    
  double get_new_log_evalue(const subsequence& s) {

    double score = min_score; 
    vector<int>::iterator it = (*sequences)[s.first].begin()+s.second;
    for(int i = 0; i < minL; i++, it++)
      score += evaluator.Iupd[*it][profile[i][*it]];

    double log_pvalue = evaluator.get_pvalue(0, score);
    int Lindex = 1;
    for(int i = minL; i < maxL; i++, it++, Lindex++) {
      
      score += scores[i] + evaluator.Iupd[*it][profile[i][*it]];
      log_pvalue = MIN(log_pvalue, evaluator.get_pvalue(Lindex, score));
    }
    
     return log_pvalue;
  }

  double get_checked_new_log_evalue(const subsequence& s, vector<double>& new_log_alignment_count) {

    double score = min_score; 
    vector<int>::iterator it = (*sequences)[s.first].begin()+s.second;
    for(int i = 0; i < minL; i++, it++)
      score += evaluator.Iupd[*it][profile[i][*it]];

    op_pos = 0; op_score = lower_bounds[0];
    log_evalue = (score > lower_bounds[0] ? op_score = score, evaluator.get_pvalue(0, score) : BIG_EVALUE)
      + new_log_alignment_count[0];
      
    int Lindex = 1;
    for(int i = minL; i < maxL; i++, it++, Lindex++) {
      
      score += scores[i] + evaluator.Iupd[*it][profile[i][*it]];
      if(score > lower_bounds[Lindex]) {
	double curr_log_evalue = evaluator.get_pvalue(Lindex, score) + new_log_alignment_count[Lindex];
	if(log_evalue > curr_log_evalue) {
	  log_evalue = curr_log_evalue;
	  op_score = score; op_pos = Lindex;
	}
      }
    }

    return log_evalue;
  }

  double get_new_score(const subsequence& s) {

    double score = min_score; 
    vector<int>::iterator it = (*sequences)[s.first].begin()+s.second;
    for(int i = 0; i < minL; i++, it++)
      score += evaluator.Iupd[*it][profile[i][*it]];

    return score;
  }

  void print(vector<string>& names) {

    set_profile();
    update_evalue();

#if DEBUG
    
    for(int i = 0; i < K; i++) {
      for(int j = 0; j < maxL; j++)
	cout << profile[j][i] << " ";
      cout << endl;
    }
    cout << endl;

    cout << "log(e-value): " << log_evalue 
#if FIXED
	 << " log(count): " << log_alignment_count << endl;
#else
	 << " log(count): " << log_alignment_count[op_pos] << endl;
#endif

    cout << "Optimal score: " << op_score << endl;
          
#endif

#if !FIXED 
    cout << "Width: " << op_pos+minL << endl;
    //cout << log_evalue << endl; 
#endif

    for(int i = 0; i < starting_positions.size(); i++) {
      cout << names[i] << " " << starting_positions[i]+1 << " ";
      for(int j = 0; j < op_pos+minL; j++)
	cout << numberToChar((*sequences)[i][starting_positions[i]+j]) ;

      cout << endl;
    }
  }
};

motif_evaluator motif::evaluator;
int motif::K = 4; int motif::minL = 5; int motif::maxL = 30; int motif::max_minus_minL = 25;

#if FIXED
double* motif::log_lengths;
#else
double** motif::log_lengths;
#endif

vector<vector<int> >* motif::sequences;
int** motif::profile = 0;

deque<double> motif::scores;
double motif::op_score; int motif::op_pos;
vector<double> motif::lower_bounds;
double motif::nlogn;

class motif_ptr {

private:
  
  motif* p;

public:

  motif_ptr() {}
  motif_ptr(motif* given_p) :  p(given_p) { }
  bool operator<(const motif_ptr& given_p) const { return (*p) < (*given_p.p); }
  bool operator<(double given_log_evalue ) const { return (*p) < given_log_evalue; }
  bool operator==(const motif_ptr& given_p) const { return (*p) == (*given_p.p); }
  motif* operator->() { return p; }
  motif& operator*() { return *p; }
  operator motif*() const { return p; } 

};

template<class T> class bounded_priority_queue : public priority_queue<T> {

private:
  int max_size;

public:

  explicit bounded_priority_queue(int given_max_size) { max_size = given_max_size; }

  void pop() {

    delete(priority_queue<T>::top());
    priority_queue<T>::pop();
  }

  bool can_push(double priority) {

    return (priority_queue<T>::size() < max_size || 
	    priority_queue<T>::top() < priority);
  }

  void unchecked_push(const T& x) {

      priority_queue<T>::push(x);
      if(priority_queue<T>::size() > max_size)
	pop();
  }

  void push(const T& x) { 

    if(priority_queue<T>::size() < max_size || 
       x < priority_queue<T>::top()) 
      unchecked_push(x); 
  }

  void remove_duplicates() {

    vector<T> temp;
    while(!priority_queue<T>::empty()) {
      T curr = priority_queue<T>::top(); priority_queue<T>::pop();
      temp.push_back(curr);
    }
    
    typename vector<T>::iterator first = temp.begin();
    while(first != temp.end()) {

      bool dup = false;
      for(typename vector<T>::iterator next = first+1;
	  next != temp.end() && !(*first < *next); next++)
	if(*first == *next) {
	  dup = true; break;
	}

      if(!dup)
	priority_queue<T>::push(*first);

      first++;
    }	      
  }

  void print() {

    vector<T>& c = priority_queue<T>::c;
    for(typename vector<T>::iterator it = c.begin(); it != c.end(); it++)
      it->print();
  }
};

int main(int argc, char** argv) {

  motif::K = 4; double pu[motif::K];
  for(int i = 0; i < motif::K; i++)
    pu[i] = 0;

  motif::minL = atoi(argv[3]); 
  motif::maxL = atoi(argc > 4 ? argv[4] : argv[3]); 
  motif::max_minus_minL = motif::maxL - motif::minL;

  vector<vector<int> > sequences; motif::sequences = &sequences;
  vector<string> names;
  read_sequences(argv[1], sequences, names, pu);
  int max_size = atoi(argv[2]);
  
  double sum = 0;
  for(int i = 0; i < motif::K; i++)
    sum += pu[i];

  for(int i = 0; i < motif::K; i++)
    pu[i] /= sum; 

  int N = motif::sequences->size(); int Q = RND_UP_TO_POWER_2(N*50);

#if DEBUG
  cout << "pu: ";
  for(int i = 0; i < motif::K; i++)
    cout << pu[i] << " "; 
  cout << endl;
  cout << "N = " << N << ", Q = " << Q << endl;
#endif
  
#if FIXED
  motif::log_lengths = new double[N];
  for(int i = 0; i < N; i++) 
    motif::log_lengths[i] = log((int) sequences[i].size()-motif::minL);
#else
  motif::log_lengths = new double*[N];
  for(int i = 0; i < N; i++) {
    motif::log_lengths[i] = new double[motif::max_minus_minL+1];
    for(int L = motif::minL; L <= motif::maxL; L++) 
      motif::log_lengths[i][L-motif::minL] = log((int) sequences[i].size()-L);
  }
#endif

  motif::evaluator.init(N, motif::K, motif::minL, motif::maxL, pu, Q);
  motif::evaluator.setN(2); motif::nlogn = -2*log(2);

#if DEBUG
  cout << "Evaluator initialized" << endl;
#endif

#if FIXED
  bounded_priority_queue<motif_ptr> top_motifs(max_size);
#else
  int factor = (argc > 5 ? atoi(argv[5]) : 5);
  bounded_priority_queue<motif_ptr> top_motifs(factor*max_size);
  //bounded_priority_queue<motif_ptr> top_motifs((N-1)*max_size);
#endif

#if FIXED
  motif::lower_bounds.resize(motif::max_minus_minL+1);
  for(int k = 0; k < motif::max_minus_minL+1; k++)
	motif::lower_bounds[k] = 0;
#endif

  for(int i = 0; i < N; i++)
    for(int j = i+1; j < N; j++) {

#if !FIXED
      motif::lower_bounds.resize(motif::max_minus_minL+1);
      for(int k = 0; k < motif::max_minus_minL+1; k++)
	motif::lower_bounds[k] = 0;
#endif
      
      for(int k = 0; k <= ((int) sequences[i].size()) - motif::maxL; k++) {
	motif m(subsequence(i, k), subsequence(j, 0));
	vector<int>::iterator it1 = sequences[i].begin()+k+motif::maxL;
	vector<int>::iterator it2 = sequences[j].begin()+motif::maxL;
	vector<int>::iterator it1_start = it1;

	do {

	  if(top_motifs.can_push(m.get_log_evalue()))
	    top_motifs.unchecked_push(motif_ptr(new motif(m, i, j, it1-it1_start)));
	  else 
	    motif::lower_bounds[motif::op_pos] = motif::op_score;

	  if(it1 == sequences[i].end() || it2 == sequences[j].end())
	    break;

	  m.increment(i, j, *it1, *it2); 
	  //m.increment(i, j, *(it1-motif::max_minus_minL), *(it2-motif::max_minus_minL)); 
	  it1++; it2++;
	} while(1);
      }

      for(int k = 0; k <= ((int) sequences[j].size()) - motif::maxL; k++) {
	motif m(subsequence(i, 1), subsequence(j, k));
	vector<int>::iterator it1 = sequences[i].begin()+1+motif::maxL;
	vector<int>::iterator it2 = sequences[j].begin()+k+motif::maxL;
	vector<int>::iterator it1_start = it1;

	do {

	  if(top_motifs.can_push(m.get_log_evalue()))
	    top_motifs.unchecked_push(motif_ptr(new motif(m, i, j, it1-it1_start)));
	  else 
	    motif::lower_bounds[motif::op_pos] = motif::op_score;

	  if(it1 == sequences[i].end() || it2 == sequences[j].end())
	    break;

	  m.increment(i, j, *it1, *it2); 
	  //m.increment(i, j, *(it1-motif::max_minus_minL), *(it2-motif::max_minus_minL)); 
	  it1++; it2++;
	} while(1);
      }
    }

#if DEBUG
  cout << "Initialized" << endl;
#endif

  bounded_priority_queue<motif_ptr> new_top_motifs(max_size);
  bounded_priority_queue<motif_ptr>* current = &top_motifs;
  bounded_priority_queue<motif_ptr>* next = &new_top_motifs;

  for(int n = 3; n <= N; n++) {
    
    //bounded_priority_queue<motif_ptr>* next = new bounded_priority_queue<motif_ptr>((N-n+1)*max_size);

    /*    
    //cout << "Stage " << n << ": ";
    while(current->size() > 0) {
      motif_ptr x = current->top();
      //cout << x->get_log_evalue() << ",";
      //for(int j = 0; j < motif::max_minus_minL+1; j++)
      //	motif::lower_bounds[j] = 0;
      //x->set_profile(); x->update_evalue();
      //cout << motif::op_pos+motif::minL << "," << x->get_log_evalue() << " ";
      //cout << motif::op_pos << ",";
      //if(current->size() == 1)
      //cout << x->get_log_evalue();

      next->push(new motif(*x)); current->pop(); 
    } //cout << endl << endl; swap(current, next);
    */

    motif::evaluator.setN(n); motif::nlogn = -n*log(n);

    while(!current->empty()) {
      motif_ptr p = current->top();
      p->set_profile();

#if !PR1
      motif* best_child = 0; 
      double best_evalue = BIG_EVALUE; // Make a better choice here
#endif

      map<int, int>::iterator it = p->get_starting_positions().begin();
      for(int i = 0; i < N; i++) {

	while(it->first < i && it != p->get_starting_positions().end()) it++;
	if(it->first != i) {
	  
#if FIXED
	  double new_log_alignment_count = p->get_log_alignment_count() + motif::log_lengths[i];
#else
	  vector<double> new_log_alignment_count(motif::max_minus_minL+1);
	  for(int L_minus_min = 0; L_minus_min <= motif::max_minus_minL; L_minus_min++)
	    new_log_alignment_count[L_minus_min] = p->get_log_alignment_count(L_minus_min) + motif::log_lengths[i][L_minus_min];
#endif

#if FIXED
	  double score_lower_bound = 0;
#else
	  for(int j = 0; j < motif::max_minus_minL+1; j++)
	    motif::lower_bounds[j] = 0;
#endif

	  for(int j = 0; j <= (int) sequences[i].size()-motif::maxL; j++) {

#if FIXED	    
	    double new_score = p->get_new_score(subsequence(i, j)); 
	    if(new_score > score_lower_bound) {

#if CONS
	      double log_evalue = -new_score;
#else
	      double log_evalue = motif::evaluator.get_pvalue(0, new_score) + new_log_alignment_count;
#endif

#else
	      //double log_evalue = p->get_new_log_evalue(subsequence(i, j)) + new_log_alignment_count;
	      double log_evalue = p->get_checked_new_log_evalue(subsequence(i, j), new_log_alignment_count);
#endif

#if PR1
	      if(next->can_push(log_evalue))
		next->unchecked_push(motif_ptr(new motif(*p, subsequence(i, j), log_evalue)));
#else
	      if(log_evalue < best_evalue) {
		//delete(best_child);
		best_child = new motif(*p, subsequence(i, j), log_evalue);		
		best_evalue = log_evalue;
	      }
#if FIXED
	      else
		score_lower_bound = new_score;
	    }
#else
	      else 
		motif::lower_bounds[motif::op_pos] = motif::op_score;
#endif

#endif
	  }
	}
      }	  

#if !PR1
      if(best_evalue < BIG_EVALUE)
	next->unchecked_push(best_child);
#endif
      current->pop();
    }

    next->remove_duplicates();

#if DEBUG
    cout << "Done with stage " << n << " " << next->size() << endl;        
    //motif_ptr p = next->top(); p->print();    
#endif

    swap(current, next);
    //current = next;
  }

#if DEBUG
  int to_print = 4;
#else
  int to_print = 1;
#endif

  while(current->size() != to_print)
    current->pop();

  for(int i = 0; i < to_print; i++) {
    motif_ptr p = current->top();
    p->print(names);
    cout << endl;
    current->pop();
  }
}
