#ifndef _ARRAY_H_
#define _ARRAY_H_

//
// Class for storing the values of the pmf
// in an array (for Hirji's algorithm). 
// The values can also be stored in a list (see list.h)
// or any other data structure which has the same public
// interface as this class.
//
// elem_t = datatype for the array elements
//
template<class elem_t>
class array_t {

 private:

  //
  // data[0..size-1] contains the data.
  // cur_i is an index for iterating over the array.
  //
  elem_t* data; int size; int cur_i;

 public:


  array_t() { size = 0; }
  void init(int given_size) { size = given_size; data = new elem_t[size]; }
  array_t(int given_size) { init(given_size); }

  inline void set(int i, elem_t value) { data[i] = value; }
  inline elem_t& operator[](int i) { return data[i]; }

  //
  // Sets the value at the current index to 0
  //
  inline void remove_cur() { data[cur_i] = 0; }

  //
  // Functions for iterating over the non-zero values of the array
  // using the variable cur_i. 
  //
  // start: Initializes cur_i and returns the first index.
  // not_last: Returns 1 if cur_i is not at the last index.
  // next: Returns the next index corresponding to a non-zero value.
  // 

  //
  // minI is the index for the first non-zero value in the array.
  //
  inline int start(int minI) { cur_i = minI; return minI; }

  //
  // maxI is the index for the last non-zero value in the array.
  //
  inline int not_last(int maxI) { return cur_i <= maxI; }

  //
  // maxI is the index for the last non-zero value in the array.
  //
  inline int next(int maxI) { while(++cur_i <= maxI) if(data[cur_i] != 0) break; return cur_i; }

  inline void destructive_copy(array_t<elem_t>& a) { 
    delete[](data); *this = a; a.data = 0; 
  }

  ~array_t() { delete[](data); }
};

#endif
