
### CUDA streams
![](../pics/cuda_streams.png)

### Asynchronous data transfer
For async transfer, the data needs to be on pinned memory so that there are no page faults. It can then be copied by using streams.

![](../pics/async_transfer.png)
![](../pics/async_transfer_2.png)
