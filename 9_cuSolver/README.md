## What is cuSolver (from the docs)
cuSolver combines three separate components under a single umbrella. The first part of cuSolver is called cuSolverDN, and deals with dense matrix factorization and solve routines such as LU, QR, SVD and LDLT, as well as useful utilities such as matrix and vector permutations.

Next, cuSolverSP provides a new set of sparse routines based on a sparse QR factorization. Not all matrices have a good sparsity pattern for parallelism in factorization, so the cuSolverSP library also provides a CPU path to handle those sequential-like matrices. For those matrices with abundant parallelism, the GPU path will deliver higher performance. The library is designed to be called from C and C++.

The final part is cuSolverRF, a sparse re-factorization package that can provide very good performance when solving a sequence of matrices where only the coefficients are changed but the sparsity pattern remains the same.

The GPU path of the cuSolver library assumes data is already in the device memory. It is the responsibility of the developer to allocate memory and to copy data between GPU memory and CPU memory using standard CUDA runtime API routines, such as cudaMalloc(), cudaFree(), cudaMemcpy(), and cudaMemcpyAsync().


## Naming convention (from the docs)
The cuSolverDN library provides two different APIs; legacy and generic.

The functions in the legacy API are available for data types float, double, cuComplex, and cuDoubleComplex. The naming convention for the legacy API is as follows:

cusolverDn<t><operation>
where <t> can be S, D, C, Z, or X, corresponding to the data types float, double, cuComplex, cuDoubleComplex, and the generic type, respectively. <operation> can be Cholesky factorization (potrf), LU with partial pivoting (getrf), QR factorization (geqrf) and Bunch-Kaufman factorization (sytrf).

The functions in the generic API provide a single entry point for each routine and support for 64-bit integers to define matrix and vector dimensions. The naming convention for the generic API is data-agnostic and is as follows:

cusolverDn<operation>
where <operation> can be Cholesky factorization (potrf), LU with partial pivoting (getrf) and QR factorization (geqrf).

The cuSolverSP library functions are available for data types float, double, cuComplex, and cuDoubleComplex. The naming convention is as follows:

cusolverSp[Host]<t>[<matrix data format>]<operation>[<output matrix data format>]<based on>
where cuSolverSp is the GPU path and cusolverSpHost is the corresponding CPU path. <t> can be S, D, C, Z, or X, corresponding to the data types float, double, cuComplex, cuDoubleComplex, and the generic type, respectively.

The <matrix data format> is csr, compressed sparse row format.

The <operation> can be ls, lsq, eig, eigs, corresponding to linear solver, least-square solver, eigenvalue solver and number of eigenvalues in a box, respectively.

The <output matrix data format> can be v or m, corresponding to a vector or a matrix.

<based on> describes which algorithm is used. For example, qr (sparse QR factorization) is used in linear solver and least-square solver.