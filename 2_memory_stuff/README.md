## CUDA memory transfer

```
cudaMemcpy(destination ptr, source ptr, size in bytes, direction)
```

Direction
- Host to device - cudamemcpyhtod
- Device to host - cudamemcpydtoh
- Device to device - cudamemcpydtod



## Data transfers
The CPU and the GPU have its own memory space with solid memory bandwidth. Obviously the weakest link is the interconnection between the CPU and the GPU. PCIE provides a bandwidth of 8GB/s in the ideal case but in practise it could be lower. 

## When GPU <-> CPU memory transfers are performed?

![](../pics/data_transfers.png)
![](../pics/data_transfers_2.png)

## Types of data transfers
![](../pics/types_data_transfers.png)

### Difference between pageable and pinned memory
When you allocate memory in host code, the allocations are pageable by default, which means it can be swapped to disk. So our data may not be present in a physical memory when the transfer to device is requested. On the other hand, pinned memory is allocated within the kernel memory space which cannot be swapped and therefor such data are always physically present in the memory. When a regular memory transfer is issued and allocation of page locked memory is necessary.

When a regular memory is requested on the host, a space is created in the pge locked memoery, memory is copied to page locked memory, the host is blocked while the transfer takes place, and once the transfer is complete, the regular memory space is deleted. This could be an overhead. However, with pinned memory the data is always accessible. 

![](../pics/transfer_rate.png)

That is why peak transfer rates with pinned memory are higher compared to transfer rates of regular memory since it avoids additional implicit memcopies within the host memory. 

Allocation is done using ```cudaMallocHost``` as shown here
![](../pics/allocation.png)

### Summary

![](../pics/summary.png)

You should not over allocate pinned memory since it reduces the amount of physical memory available for kernel/operating system. In such cases, the system could become unstable. 

### Where are local variables stored?
![](../pics/local_variables.png)

Local variables are stored in registers. It is local to each thread and is not shared with neighboring threads.

![](../pics/shared_memory.png)

### Performance of Unified Virtual Memory (introduced in CUDA 6)
Developer views the memory as if there is only one shared memory between host and device. However, at the hardware level, there are 2 distinct spaces. 

![](../pics/unified_memory.png)

![](../pics/uvm_code.png)

Performance is degraded as data transfer occurs over multiple small data transfers when it is requested from the host. It introduced latency and also poor bandwidth. 
![](../pics/uvm_performance.png)


### Asynchronous data transfer
For async transfer, the data needs to be on pinned memory so that there are no page faults. It can then be copied by using streams.

![](../pics/async_transfer.png)
![](../pics/async_transfer_2.png)

### CUDA streams
![](../pics/cuda_streams.png)

### Device management
![](../pics/device_management.png)


Follow this link for an explanation of memory transfers: 
[](https://www.youtube.com/watch?v=Yv4thF9tvPo&t=8s)

[](https://www.youtube.com/watch?v=0wf_y5fVbP0)

[](https://github.com/khadjiyiannakou/trainCuda)
