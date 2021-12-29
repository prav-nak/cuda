#include <iostream>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
Having inactive threads in warp will be a great waste of resource in streaming multiprocessor (SM).
*/
__global__ void print_details_of_warps(int *data, int size)
{
    // global thread id
    int gid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    // warp id: Each thread block is executed in warps (set of 32 threads. So local threadid/32)
    int warp_id = threadIdx.x / 32;

    // global block id
    int gbid = blockIdx.y * gridDim.x + blockIdx.x;

    if (gid < size)
    {
        std::cout << "local thread id = " << threadIdx.x << std::endl;
        std::cout << "block id in x = " << blockIdx.x << std::endl;
        std::cout << "block id in y = " << blockIdx.y << std::endl;
        std::cout << "block id in z = " << blockIdx.z << std::endl;
        std::cout << "global thread id = " << gid << std::endl;
        std::cout << "warp id = " << warp_id << std::endl;
        std::cout << "global block id = " << gbid << std::endl;
        std::cout << "Array value = " << data[gid] << std::endl;
        __nanosleep(1000); // sleep for 1 micro-second to see in the profiler
    }
}

int main(int argc, char **argv)
{

    int array_size = 64;
    int array_byte_size = sizeof(int) * array_size;

    // Create the host data array
    int host_data[array_size];
    for (int i = 0; i < array_size; ++i)
        host_data[i] = i;

    // Create the device data array
    int *device_data;
    cudaMalloc((void **)&device_data, array_byte_size);
    cudaMemcpy(device_data, host_data, array_byte_size, cudaMemcpyHostToDevice);

    // 1. Block size of 32
    dim3 block_size1(32);
    dim3 grid_size1(2, 2);

    print_details_of_warps<<<grid_size1, block_size1>>>(device_data, array_size);
    cudaDeviceSynchronize();

    // 2. Block size of 32
    dim3 block_size2(40);
    dim3 grid_size2(2, 2);

    print_details_of_warps<<<grid_size2, block_size2>>>(device_data, array_size);
    cudaDeviceSynchronize();

    cudaFree(device_data);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}