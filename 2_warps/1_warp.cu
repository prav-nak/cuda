#include <iostream>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
This is a one dimensional block. So threadIdx.x is sufficient. Otherwise you need to compute the id in 3D
threadId = threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z). Every 32 threads of this index is a new warp.
*/
__global__ void print_details_of_warps()
{
	// global thread id
	int gid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

	// warp id: Each thread block is executed in warps (set of 32 threads. So local threadid/32)
	int warp_id = threadIdx.x / 32;

	// global block id
	int gbid = blockIdx.y * gridDim.x + blockIdx.x;

	printf("local thread id = %d\n", threadIdx.x);
	printf("block id in x = = %d\n", blockIdx.x );
	printf("block id in y = = %d\n", blockIdx.y );
	printf("block id in z = = %d\n", blockIdx.z);
	printf("global thread id = = %d\n", gid);
	printf("warp id = = %d\n", warp_id);
	printf("global block id = = %d\n", gbid );
}

int main(int argc, char **argv)
{
	dim3 block_size(42);
	dim3 grid_size(2, 2);

	print_details_of_warps<<<grid_size, block_size>>>();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return EXIT_SUCCESS;
}