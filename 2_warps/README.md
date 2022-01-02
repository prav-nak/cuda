- [Warps](#warps)
- [On-chip components](#on-chip-components)
- [Warp divergence](#warp-divergence)
    - [if-then-else](#if-then-else)
      - [Deadlock](#deadlock)

# Warps
- A warp is a set of 32 threads within a thread block such that all the threads in a warp execute the same instruction. These threads are selected serially by the SM. Once a thread block is launched on a multiprocessor (SM), all of its warps are resident until their execution finishes.

```threadId = threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z)```. Every 32 threads of this index is a new warp.

- Number of warps per block = Block size / warp size. Here block size is the number of threads in each block and warp size is 32.

- Resource allocations like shared memory for block will be done considering number of warps

- Having inactive threads in warp will be a great waste of resource in streaming multiprocessor (SM)

# On-chip components

- CUDA cores (FP32, FP64, INT, Tensor cores)
- Shared memory and L1 cache
- Registers
- Load/Store units
- warp schedulers
- Special function units

<img src="../pics/on_chip_components.png" alt="drawing" width="400"/>

# Warp divergence
Threads from a block are bundled into fixed-size warps for execution on a CUDA core, and threads within a warp must follow the same execution trajectory. All threads must execute the same instruction at the same time. In other words, threads cannot diverge.

### if-then-else

The most common code construct that can cause thread divergence is branching for conditionals in an if-then-else statement. If some threads in a single warp evaluate to 'true' and others to 'false', then the 'true' and 'false' threads will branch to different instructions. Some threads will want proceed to the 'then' instruction, while others the 'else'.

Intuitively, we would think statements in then and else should be executed in parallel. However, because of the requirement that threads in a warp cannot diverge, this cannot happen. The CUDA platform has a workaround that fixes the problem, but has negative performance consequences.

 Intuitively, we would think statements in then and else should be executed in parallel. However, because of the requirement that threads in a warp cannot diverge, this cannot happen. The CUDA platform has a workaround that fixes the problem, but has negative performance consequences.

When executing the if-then-else statement, the CUDA platform will instruct the warp to execute the then part first, and then proceed to the else part. While executing the then part, all threads that evaluated to false (e.g. the else threads) are effectively deactivated. When execution proceeds to the else condition, the situation is reversed. As you can see, the then and else parts are not executed in parallel, but in serial. This serialization can result in a significant performance loss.

#### Deadlock
Thread divergence can also cause a program to deadlock. Consider the following example:
```
//my_Func_then and my_Func_else are some device functions
if (threadidx.x <16)
{
	myFunc_then();
	__syncthread();
}else if (threadidx >=16)
{
	myFunc_else();
	__syncthread();
}
```

The first half of the warp will execute the then part, then wait for the second half of the warp to reach __syncthread(). However, the second half of the warp did not enter the then part; therefore, the first half of the warp will be waiting for them forever. 
