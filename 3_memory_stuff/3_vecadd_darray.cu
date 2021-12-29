#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cassert>
#include <iostream>
#include "common_routines.h"
#include "d_array.h"

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
    float *a, *b, *c, *h_verify;

    a = (float*)malloc(num_bytes);
    b = (float*)malloc(num_bytes);
    c = (float*)malloc(num_bytes);
    h_verify = (float*)malloc(num_bytes);

    // Initialize vectors to random numbers
    init_random_vec(a, N, 1.0f, 10.0f);
    init_random_vec(b, N, 1.0f, 10.0f);  

    // Add the 2 vectors on CPU
    vecadd_cpu(a, b, c, N);

    // Device vectors
    d_array<float> d_a(N);
    d_array<float> d_b(N);
    d_array<float> d_c(N);

    // Copy vectors from host to device
    d_a.copy_host_to_device(a, N);
    d_b.copy_host_to_device(b, N);

    // Add vectors on GPU
    vecadd_gpu<<<1, 5>>>(d_a.getData(), d_b.getData(), d_c.getData(), N);
    cudaDeviceSynchronize();

    // Copy the result vector from GPU to CPU
    d_c.copy_device_to_host(h_verify, N);

    //print_vector(c, N);
    //print_vector(h_verify, N);

    // Check if the CPU result is the same as the GPU result
    check_equality(c, h_verify, N);

    // Free the resources
    free(a);
    free(b);
    free(c);
    free(h_verify);

    return 0;
}