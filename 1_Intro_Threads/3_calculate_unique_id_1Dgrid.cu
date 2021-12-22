#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void unique_globalid_1Darray_calc(int* input){

	int threadId = threadIdx.x;
	int offset = blockIdx.x * blockDim.x;
	int globalId = threadId + offset;

	printf("value of the array at index %d is %d\n", globalId, input[globalId]);
}

int main() {

	int nx, ny, nz;
	int array_size = 8;

	nx = 2;
	ny = 3;
	nz = 1;
	
	int array_byte_size = sizeof(int) * array_size;

	dim3 grid_size(2,3,4);
    dim3 block_size(nx,ny,nz);

	// Create the host data array
	int host_data[array_size];
	for (int i =0; i<array_size; ++i)
		host_data[i] = i;

	// Create the device data array
	int * device_data;
	cudaMalloc((void**)&device_data, array_byte_size);
	cudaMemcpy(device_data, host_data, array_byte_size, cudaMemcpyHostToDevice);

	// Get the array components using device
	unique_globalid_1Darray_calc<<<1, array_size>>>(device_data);

	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}
