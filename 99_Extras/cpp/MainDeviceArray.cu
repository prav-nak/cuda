#include "d_array.h"
#include <iostream>

/*
    1. Create a float array on the host
    2. Copy the array to device
    3. Copy the device array back to host into a different array
    4. Print to validate
*/
int main(){

    d_array<float> test(10);
    float* h_vec_src;
    float* h_vec_dest;
    
    h_vec_src = (float *) malloc(test.getSize());
    h_vec_dest = (float *) malloc(test.getSize());

    for (int i=0; i<test.getSize(); ++i){
        h_vec_src[i] = i*10;
    }

    test.copy_host_to_device(h_vec_src, test.getSize());
    test.copy_device_to_host(h_vec_dest, test.getSize());

    for (int i=0; i<10; ++i){
        std::cout<<h_vec_dest[i]<<"\n";
    }
    free(h_vec_src);
    free(h_vec_dest);

    return 0;
}