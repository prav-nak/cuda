#include "cuda_runtime.h"

/*
  This kernel has a branching over warp ids. 
  Since all threads in a warp run simultaneously, 
  there is no warp divergence. 
*/
__global__ void kernel_without_divergence()
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;

	float a, b;
	int warp_id = globalId / 32;
	if (warp_id % 2 == 0) {
		a = 1.;
		b = 2.;
	}
	else{
		a = 2.;
		b = 1.;
    }
    a++;
    b++;
}

/*
  This kernel has a branching over global ids. 
  Hence, there is warp divergence. However, the compiler 
  when run with optimizations, can try to fix it.
*/
__global__ void kernel_with_divergence()
{
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	float a, b;

	if (globalId % 2 == 0) {
		a = 1.;
		b = 2.;
	}
	else {
		a = 2.;
		b = 1.;
    }
    a++;
    b++;
}

int main(int argc, char **argv)
{
	int size = 1 << 10;

	dim3 block_size(128);
	dim3 grid_size((size + block_size.x - 1) / block_size.x);

	kernel_without_divergence<<<grid_size, block_size>>>();
	cudaDeviceSynchronize();

	kernel_with_divergence<<<grid_size, block_size>>>();
	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}