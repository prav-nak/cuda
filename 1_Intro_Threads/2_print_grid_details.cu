#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// Function to print grid information
__global__ void print_threadIds()
{
	printf("gridDim.x: %d, gridDim.y: %d, gridDim.z:%d, blockId.x: %d, blockId.y: %d, blockId.z:%d, blockDim.x: %d, blockDim.y: %d, blockDim.z: %d\n", gridDim.x, gridDim.y, gridDim.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z);
}

int main()
{

	int nx, ny, nz;

	nx = 2;
	ny = 3;
	nz = 1;

	dim3 grid_size(2, 3, 4);
	dim3 block_size(nx, ny, nz);

	print_threadIds<<<grid_size, block_size>>>();

	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}
