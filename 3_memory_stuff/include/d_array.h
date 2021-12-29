#ifndef _D_ARRAY_H_
#define _D_ARRAY_H_

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

/*
A basic device array class. 
This class provides functions to 
	1. allocate an array on the device
	2. resize an array to required size
	3. return the size of an array
	4. Templated over the datatype
	5. Set function to copy a host array into a device array
	5. Get function to copy a device array back into the host
More details at https://www.quantstart.com/articles/dev_array_A_Useful_Array_Class_for_CUDA/ 
*/
template <class T>
class d_array
{
	// public functions
	public:
		explicit d_array():start_(0),end_(0) {}

		// constructor
		explicit d_array(size_t size) {
			allocate(size);
		}

		// destructor
		~d_array() {
			free();
		}

		// resize the vector
		void resize(size_t size) {
			free();
			allocate(size);
		}

		// get the size of the array
		size_t getSize() const {
			return end_ - start_;
		}

		// get data
		const T* getData() const {
			return start_;
		}

		T* getData() {
			return start_;
		}

		// copy host to device
		void copy_host_to_device(const T* src, size_t size) {
			size_t min = std::min(size, getSize());
			cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);
			if (result != cudaSuccess) {
				throw std::runtime_error("failed to copy to device memory");
			}
		}

		// copy device to host
		void copy_device_to_host(T* dest, size_t size) {
			size_t min = std::min(size, getSize());
			cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
			if (result != cudaSuccess) {
				throw std::runtime_error("failed to copy to host memory");
			}
		}


	// private functions
	private:
		// allocate memory on the device
		void allocate(size_t size) {
			cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(T));
			if (result != cudaSuccess) {
				start_ = end_ = 0;
				throw std::runtime_error("failed to allocate device memory");
			}
			end_ = start_ + size;
		}

		// free memory on the device
		void free() {
			if (start_ != 0) {
				cudaFree(start_);
				start_ = end_ = 0;
			}
		}

		T* start_;
		T* end_;
};

#endif
