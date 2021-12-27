#ifndef __COMPUTE_CAP_H
#define __COMPUTE_CAP_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <iostream>

/*
#--------------------------------------------------------------------------;
#    Re-usable class to return device properties and compute capabilities. ;
#--------------------------------------------------------------------------;
*/
class ComputeCapability{
    public:
        ComputeCapability(){
        int iDev = 0;
        cudaDeviceProp iProp;

        cudaGetDeviceProperties(&iProp, iDev);
        
        _devId = iDev;
        _device_name = std::string(iProp.name);
        _multiProcessorCount = iProp.multiProcessorCount;
        _major = iProp.major;
        _minor = iProp.minor;
        _totalGlobalMem_KB = iProp.totalGlobalMem/ 1024.0;
        _totalConstMem_KB = iProp.totalConstMem / 1024.0;
        _sharedMemPerBlock_KB = iProp.sharedMemPerBlock / 1024.0;
        _sharedMemPerMultiprocessor_KB = iProp.sharedMemPerMultiprocessor / 1024.0;
        _regsPerBlock = iProp.regsPerBlock;
        _warpSize = iProp.warpSize;
        _maxThreadsPerBlock = iProp.maxThreadsPerBlock;
        _maxThreadsPerMultiProcessor = iProp.maxThreadsPerMultiProcessor;
        _maxGridSize_x = iProp.maxGridSize[0];
        _maxGridSize_y = iProp.maxGridSize[1];
        _maxGridSize_z = iProp.maxGridSize[2];
        _maxThreadsDim_x = iProp.maxThreadsDim[0];
        _maxThreadsDim_y = iProp.maxThreadsDim[1];
        _maxThreadsDim_z = iProp.maxThreadsDim[2];
    }

    void print_details(){
        std::cout<<"Device "<< _devId<<": "<< _device_name << std::endl;
        std::cout<<"Number of multiprocessors: "<< _multiProcessorCount<<std::endl;
        std::cout<<"  Compute capability: "<< _major<<", "<<_minor <<std::endl;
        std::cout<<"  Total amount of global memory (KB): "<< _totalGlobalMem_KB<<std::endl;
        std::cout<<"  Total amount of constant memory (KB): "<< _totalConstMem_KB <<std::endl;
        std::cout<<"  Total amount of shared memory per block (KB): "<< _sharedMemPerBlock_KB <<std::endl;
        std::cout<<"  Total amount of shared memory per MP (KB): "<< _sharedMemPerMultiprocessor_KB <<std::endl;
        std::cout<<"  Total number of registers available per block: "<< _regsPerBlock<<std::endl;
        std::cout<<"  Warp size: "<< _warpSize <<std::endl;
        std::cout<<"  Maximum number of threads per block:  "<< _maxThreadsPerBlock<<std::endl;
        std::cout<<"  Maximum number of threads per multiprocessor:  "<< _maxThreadsPerMultiProcessor<<std::endl;
        std::cout<<"  Maximum number of warps per multiprocessor: "<< _maxThreadsPerMultiProcessor / 32<<std::endl;
        std::cout<<"  Maximum Grid size:  "<< _maxGridSize_x <<", "<< _maxGridSize_y<<", "<< _maxGridSize_z<<std::endl;
        std::cout<<"  Maximum block dimension: "<< _maxThreadsDim_x<<", "<< _maxThreadsDim_y <<", "<< _maxThreadsDim_z<<std::endl;
    }

    private:
        int _devId;
        std::string _device_name;
        int _major, _minor;
        int _maxGridSize_x, _maxGridSize_y, _maxGridSize_z;
        int _maxThreadsDim_x, _maxThreadsDim_y, _maxThreadsDim_z;
        int _multiProcessorCount, _regsPerBlock, _warpSize;
        int _maxThreadsPerBlock, _maxThreadsPerMultiProcessor;
        float _totalGlobalMem_KB, _totalConstMem_KB, _sharedMemPerBlock_KB;
        float _sharedMemPerMultiprocessor_KB;
};

#endif