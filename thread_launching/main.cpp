#include <iostream>
#include <assert.h>
#include <chrono>
#include <thread>
using namespace std;
using namespace std::chrono;

#define NUM_OF_THREADS 8
typedef double reduce_type;

void reference_reduction(reduce_type *matrix1, reduce_type *matrix2, reduce_type *result_matrix, int size) {
  for (int i = 1; i < size; i++) {
      result_matrix[i] = (matrix1[i] + matrix2[i]);
  }
}

#define SIZE (1024)

int main() {
  thread thread_ar_round_robin[NUM_OF_THREADS];

  reduce_type *matrix1, *matrix2, *result_matrix;
  
  matrix1 = (reduce_type *) malloc(SIZE * sizeof(reduce_type));
  matrix2 = (reduce_type *) malloc(SIZE * sizeof(reduce_type));
  result_matrix = (reduce_type *) malloc(SIZE * sizeof(reduce_type));

  for(int i = 0; i < SIZE; i++){
      matrix1[i] =  (rand() % 10);
      matrix2[i] =  (rand() % 10);
  }

  auto ref_start = high_resolution_clock::now();
  for (int i = 0; i < NUM_OF_THREADS; i++) {
      thread_ar_round_robin[i] = thread(reference_reduction, matrix1, matrix2, result_matrix, SIZE);
  }

  for (int i = 0; i < NUM_OF_THREADS; i++) {
      thread_ar_round_robin[i].join();
  }
  
  auto ref_stop = high_resolution_clock::now();
  auto ref_duration = duration_cast<nanoseconds>(ref_stop - ref_start);
  double ref_seconds = ref_duration.count()/1000000000.0;

  cout << "loop time: " << ref_seconds << endl; 

}
