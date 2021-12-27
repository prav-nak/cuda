#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "compute_capability.h"

int main(){
    //query_device();
	ComputeCapability obj;
	obj.print_details();
}
