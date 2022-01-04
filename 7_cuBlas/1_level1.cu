#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <iostream>

int main(){

    cublasHandle_t handle;

    float* h_vec;
    float* d_vec;
    float* d_vec_y;
    int  result;
    float f_result;
    int SIZE=10;

    cublasCreate(&handle);

    // Create host and device vectors
    h_vec = (float*)malloc(SIZE*sizeof(float));
    for(int i=0; i<SIZE; ++i){
        h_vec[i] = i;
    }
    cudaMalloc((void **)&d_vec, SIZE*sizeof(float));
    cudaMemcpy(d_vec, h_vec, SIZE*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_vec_y, SIZE*sizeof(float));
    cudaMemcpy(d_vec_y, h_vec, SIZE*sizeof(float), cudaMemcpyHostToDevice);

    // Find the index of the largest element (1-index based)
    cublasIsamax(handle, SIZE, d_vec, 1, &result);
    std::cout<<"Index of the largest element is (1-index based): "<<result<<std::endl;

    // Find the index of the largest element (1-index based)
    cublasIsamin(handle, SIZE, d_vec, 1, &result);
    std::cout<<"Index of the smallest element is (1-index based): "<<result<<std::endl;
    
    // Sum of all the elements of the vector
    cublasSasum(handle, SIZE, d_vec, 1, &f_result);
    std::cout<<"Sum of all the elements of the vector: "<<f_result<<std::endl;

    // Multiply the vector x by the scalar α and adds it to the vector y overwriting the latest vector with the result.
    float alpha = 2;
    cublasSaxpy(handle, SIZE, &alpha, d_vec, 1, d_vec_y, 1);

    // This function copies the vector x into the vector y.
    cublasScopy(handle, SIZE, d_vec, 1, d_vec_y, 1);

    // This function computes the dot product of vectors x and y.
    cublasSdot (handle, SIZE, d_vec, 1, d_vec_y, 1, &f_result);
    std::cout<<"Dot product of the 2 vectors: "<<f_result<<std::endl;

    // Euclidean norm of the vector
    cublasSnrm2(handle, SIZE, d_vec, 1, &f_result);
    std::cout<<"Norm of the vector: "<<f_result<<std::endl;

    // This function scales the vector x by the scalar alpha and overwrites it with the result.
    alpha = 2.0;
    cublasSscal(handle, SIZE, &alpha, d_vec, 1);

    free(h_vec);
    cudaFree(d_vec);
    cublasDestroy(handle);

    return 0;
}