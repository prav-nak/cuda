#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// For a 1D grid, print the unique ID of the thread
__global__ void unique_globalid_1Darray_calc(int *input)
{
	int threadId = threadIdx.x;
	int offset = blockIdx.x * blockDim.x;
	int globalId = threadId + offset;

	printf("value of the array at index %d is %d\n", globalId, input[globalId]);
}

int main()
{
	int array_size = 8;

	int array_byte_size = sizeof(int) * array_size;

	// Create the host data array
	int host_data[array_size];
	for (int i = 0; i < array_size; ++i)
		host_data[i] = i;

	// Create the device data array
	int *device_data;
	cudaMalloc((void **)&device_data, array_byte_size);
	cudaMemcpy(device_data, host_data, array_byte_size, cudaMemcpyHostToDevice);

	// Get the array components using device
	unique_globalid_1Darray_calc<<<1, array_size>>>(device_data);

	cudaDeviceSynchronize();
	cudaDeviceReset();
	return 0;
}
