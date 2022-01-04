
- The default stream is special
    - Default stream overlap with non-default stream work cannot occur
    - There can be no execution in any non-default streams at the same time as the execution in the default stream
    - The default stream will wait for all non-default stream execution to complete before it can begin execution in the default stream
    - Operations in the default stream must complete before operations in the non-default streams can begin

### CUDA streams
![](../pics/cuda_streams.png)

### Asynchronous data transfer
For async transfer, the data needs to be on pinned memory so that there are no page faults. It can then be copied by using streams.

![](../pics/async_transfer.png)
![](../pics/async_transfer_2.png)
