#include <random>
#include <cassert>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

enum Device { cpu, gpu };

int main(){

    int nrows=10;
    int ncols=10;

    int array_size = nrows * ncols;
    int array_byte_size = sizeof(float) * array_size;

    // Host data
    float * host_mat = (float *) malloc(array_byte_size);
    for (int i=0; i<array_size; ++i){
        host_mat[i] = 1.0;
    }
    for (int i=0; i<array_size; ++i){
        std::cout<<host_mat[i]<<" ";
    }
    std::cout<<"\n=============================\n";

    // Device
    float * device_mat;
    cudaMalloc((void**)(&device_mat), array_byte_size);
    cudaMemcpy(device_mat, host_mat, sizeof(float) * array_size, cudaMemcpyHostToDevice);

    float * host_mat_2 = (float *) malloc(array_byte_size);
    cudaMemcpy(host_mat_2, device_mat, sizeof(float) * array_size, cudaMemcpyDeviceToHost);


    for (int i=0; i<array_size; ++i){
        std::cout<<host_mat_2[i]<<" ";
    }

    std::cout<<"\n=============================\n";


    free(host_mat);
    free(host_mat_2);
    cudaFree(device_mat);

    return 0;
}



// int main(void)
// {
// 	float *a_h, *b_h; // host data
// 	float *a_d, *b_d; // device data

// 	int N = 14, nBytes, i ;
// 	nBytes = N*sizeof(float);
// 	a_h = (float *)malloc(nBytes);
// 	b_h = (float *)malloc(nBytes);
// 	cudaMalloc((void **) &a_d, nBytes);
// 	cudaMalloc((void **) &b_d, nBytes);
// 	for (i=0; i<N; i++) {
// 		a_h[i] = 100.f + i;
// 	}
// 	cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);
// 	cudaMemcpy(b_d, a_d, nBytes, cudaMemcpyDeviceToDevice);
// 	cudaMemcpy(b_h, b_d, nBytes, cudaMemcpyDeviceToHost);
// 	for (i=0; i< N; i++) assert( a_h[i] == b_h[i] );
// 	free(a_h); free(b_h); cudaFree(a_d); cudaFree(b_d);
// 	return 0;
// }
