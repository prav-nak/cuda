#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cassert>
#include <iostream>
#include "common_routines.h"

// Add vectors on the CPU
void vecadd_cpu(float*a, float* b, float* c, int N){
    for(int i=0; i<N; ++i){
        c[i] = a[i] + b[i];
    }
}

// Add vectors on the GPU
__global__ void vecadd_gpu(float *a, float *b, float *c, int N){
    
    int total_threads = blockDim.x * gridDim.x;
    int n_strides = int(N / total_threads) + 1;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    for (int j=0; j<n_strides; ++j){
        if (i < N){
            c[i+j*n_strides] = a[i+j*n_strides] + b[i+j*n_strides];
        }
    }
}

int main(){

    int N = 10;
    size_t num_bytes = N * sizeof(float);

    // Host vectors: allocate
    float *a, *b, *c;

	cudaMallocManaged((void **)&a, num_bytes);
	cudaMallocManaged((void **)&b, num_bytes);

    c = (float*)malloc(num_bytes);

    // Initialize vectors to random numbers
    init_random_vec(a, N, 1.0f, 10.0f);
    init_random_vec(b, N, 1.0f, 10.0f);  

    // Add the 2 vectors on CPU
    vecadd_cpu(a, b, c, N);

    // Device vectors
    float *d_c;
    cudaMallocManaged((void**)&d_c, num_bytes);

    // Add vectors on GPU
    vecadd_gpu<<<1, 5>>>(a, b, d_c, N);
    cudaDeviceSynchronize();

    //print_vector(c, N);
    //print_vector(d_c, N);

    // Check if the CPU result is the same as the GPU result
    check_equality(c, d_c, N);

    // Free the resources
    cudaFree(a);
    cudaFree(b);
    free(c);

    cudaFree(d_c);

    return 0;
}