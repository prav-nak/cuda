## Level 1 
- Level-1 Basic Linear Algebra Subprograms (BLAS1) functions perform scalar and vector based operations.

This function finds the (smallest) index of the element of the maximum magnitude. 
```
cublasStatus_t cublasIsamax(cublasHandle_t handle, int n,
                            const float *x, int incx, int *result)
```

This function finds the (smallest) index of the element of the minimum magnitude.
```
cublasStatus_t cublasIsamin(cublasHandle_t handle, int n,
                            const float *x, int incx, int *result)
```

This function computes the sum of the absolute values of the elements of vector x. 
```
cublasStatus_t  cublasSasum(cublasHandle_t handle, int n,
                            const float           *x, int incx, float  *result)
```

This function multiplies the vector x by the scalar alpha and adds it to the vector y overwriting the latest vector with the result.
```
cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n,
                           const float           *alpha,
                           const float           *x, int incx,
                           float                 *y, int incy)
```

This function copies the vector x into the vector y.
```
cublasStatus_t cublasScopy(cublasHandle_t handle, int n,
                           const float           *x, int incx,
                           float                 *y, int incy)
```

This function computes the dot product of vectors x and y.
```
cublasStatus_t cublasSdot (cublasHandle_t handle, int n,
                           const float           *x, int incx,
                           const float           *y, int incy,
                           float           *result)
```

This function computes the Euclidean norm of the vector x.
```
cublasStatus_t  cublasSnrm2(cublasHandle_t handle, int n,
                            const float           *x, int incx, float  *result)
```


This function scales the vector x by the scalar alpha and overwrites it with the result.
```
cublasStatus_t  cublasSscal(cublasHandle_t handle, int n,
                            const float           *alpha,
                            float           *x, int incx)
```

## Level 2

## Level 3


## Matrix-matrix multiplication
- Matrices are stored in column major format