#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// For a 2D grid, print the unique ID of the thread
__global__ void unique_globalid_2Darray_calc(int *input)
{
    int nrows_to_skip = blockIdx.y;
    int nthreads_in_each_block = blockDim.x * blockDim.y;
    int n_ind_for_skipped_rows = nrows_to_skip * nthreads_in_each_block * gridDim.y;
    
    int nblocks_to_skip_eachrow = blockIdx.x;
    int n_ind_for_skipped_blocks = nblocks_to_skip_eachrow * nthreads_in_each_block;

    int local_threadid = threadIdx.y * blockDim.x + threadIdx.x;
    int gid = n_ind_for_skipped_rows + n_ind_for_skipped_blocks + local_threadid;

    printf("nrows_to_skip: %d, n_ind_for_skipped_rows: %d, nblocks_to_skip_eachrow: %d, n_ind_for_skipped_blocks: %d, local_threadid: %d, gid: %d\n", nrows_to_skip, n_ind_for_skipped_rows, nblocks_to_skip_eachrow, n_ind_for_skipped_blocks, local_threadid, gid);
    printf("value of the array at index %d is %d\n", gid, input[gid]);
}

int main()
{
    int array_size = 16;

    int array_byte_size = sizeof(int) * array_size;

    // Create the host data array
    int host_data[array_size];
    for (int i = 0; i < array_size; ++i)
        host_data[i] = i;

    // Create the device data array
    int *device_data;
    cudaMalloc((void **)&device_data, array_byte_size);
    cudaMemcpy(device_data, host_data, array_byte_size, cudaMemcpyHostToDevice);

    dim3 grid_size(2, 2, 1);
    dim3 block_size(2, 2, 1);

    // Get the array components using device
    unique_globalid_2Darray_calc<<<grid_size, block_size>>>(device_data);

    cudaFree(device_data);

    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
