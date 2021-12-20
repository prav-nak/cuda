#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_threadIds(){

	printf("threadIdx.x: %d, threadIdx.y: %d, threadIdx.z:%d\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {

	int nx, ny, nz;

	nx = 2;
	ny = 3;
	nz = 1;

	dim3 grid_size(1,1,1);
        dim3 block_size(nx,ny,nz);

	print_threadIds<<<grid_size, block_size>>>();

	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}
