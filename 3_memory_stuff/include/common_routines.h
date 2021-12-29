#include <iostream>
#include <random>

#define EPSILON 1e-8
/*
//----------------------------------------------------------;
// Print an array given a pointer to the array and its size. ;
//----------------------------------------------------------;
*/
template <class T>
void print_vector(T *a, size_t N){
    for(int i=0; i<N; ++i){
        std::cout<<a[i]<< std::endl;
    }
}

/*
//-----------------------------------------------;
// Check the equality of two arrays elementwise. ;
//-----------------------------------------------;
*/
template <class T>
void check_equality(T* a, T* b, size_t N){
    for(int i=0; i<N; ++i){
        assert(fabs(a[i] - b[i]) < EPSILON) ;
    }
}

template <class T>
void init_random_vec(T *a, int N, T lower_bound, T upper_bound) {
    for (int i=0; i<N; ++i) {
        a[i] = rand() %100;

        std::random_device rd;
        std::mt19937 mt(rd());

        if (std::is_same<T, int>::value) {
            std::uniform_int_distribution<int> dist(lower_bound, upper_bound);
            for (int i=0; i<N; ++i) {
                a[i] = dist(mt);
            }
        }
        else if (std::is_same<T, float>::value){
            std::uniform_real_distribution<float> dist(lower_bound, upper_bound);
            for (int i=0; i<N; ++i) {
                a[i] = dist(mt);
            }
        }
    }
}