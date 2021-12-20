```kernel_name<<<number_of_blocks, threads_per_block>>>(arguments()```
The arrangement of a grid is configured by these first 2 kernel launch parameters. First kernel launch paramter configures how many thread blocks in each dimension and second parameter specifies how many threads in a block in each dimention. 

There are 2 ways to specify these dimensions:
- A single integer defines in one dimension only
- A dim3 defines in 3 dimensions


```dim3 variable_name(x,y,z)```

You can access each dimension of the ```dim3``` variable by 
```
- variable_name.x
- variable_name.y
- variable_name.z
```

![](../pics/threads.jfif)


# For a given thread block, the limitation on the number of threads in each dimension are as follows:
```
- x<=1024
- y<=1024
- z<=64
- x * y * z < 1024
```

# A grid can have a maximum of following number of blocks in each dimension
```
- x < 2^32-1
- y < 65536
- z < 65536
```
