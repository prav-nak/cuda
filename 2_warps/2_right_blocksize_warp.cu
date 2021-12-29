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
        printf("local thread id = %d\n", threadIdx.x);
        printf("block id in x = %d\n", blockIdx.x );
        printf("block id in y = %d\n", blockIdx.y );
        printf("block id in z = %d\n", blockIdx.z);
        printf("global thread id = %d\n", gid);
        printf("warp id = %d\n", warp_id);
        printf("global block id = %d\n", gbid );
        printf("Array value = %d\n", data[gid]);
        //__nanosleep(1000); // sleep for 1 micro-second to see in the profiler
    }
}

int main(int argc, char **argv)
{

    int array_size = 6400;
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
    return 0;
}
