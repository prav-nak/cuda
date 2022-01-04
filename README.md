# CUDA: Table of contents
- [Know your device: Compute capability](https://github.com/prav-nak/cuda/tree/main/0_device_query)
  - Understand your device: How many SMs  do we have, how much memory we have etc.
  - What is compute capability?
  - Query device properties
- [Introduction to threads](https://github.com/prav-nak/cuda/tree/main/1_Intro_Threads)
- [Warps](https://github.com/prav-nak/cuda/tree/main/2_warps)
- [Memory](https://github.com/prav-nak/cuda/tree/main/3_memory_stuff)
- [Streams]()
  - Async operations, overlap compute with memory transfer
  - Streams can also be used to execute multiple kernels simultaneously to more fully take advantage of the device's multiprocessors
  - Timing kernels
- [Multi GPU]()
- [MPI+CUDA](https://github.com/prav-nak/cuda/blob/main/6_mpi_cuda/README.md)
    - CUDA aware MPI
    - GPUDirect: high-bandwidth, low-latency communications with NVIDIA GPUs
- [Profiling]()
