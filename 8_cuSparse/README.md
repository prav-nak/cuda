## Sparse matrix storage formats

### Coordinate list (COO)
COO stores a list of (row, column, value) tuples. Ideally, the entries are sorted first by row index and then by column index, to improve random access times.

```
cusparseStatus_t
cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr,
                  int64_t               rows, // (Host) Number of rows of the sparse matrix
                  int64_t               cols, // (Host) Number of columns of the sparse matrix
                  int64_t               nnz,  // (Host) Number of non-zero entries of the sparse matrix
                  void*                 cooRowInd, // (Device) Row indices of the sparse matrix. Array of size nnz
                  void*                 cooColInd, // (Device) Column indices of the sparse matrix. Array of size nnz
                  void*                 cooValues, // (Device) Values of the sparse martix. Array of size nnz
                  cusparseIndexType_t   cooIdxType, // (Host) Data type of cooRowInd and cooColInd
                  cusparseIndexBase_t   idxBase, // (Host) Base index of cooRowInd and cooColInd. One of CUSPARSE_INDEX_BASE_ZERO or CUSPARSE_INDEX_BASE_ONE
                  cudaDataType          valueType) // (Host) Datatype of cooValues
```

### Matrix addition
This function performs following matrix-matrix operation

```C = alpha∗A + beta∗B```

where A, B, and C are m×n sparse matrices (defined in CSR storage format by the three arrays csrValA|csrValB|csrValC, csrRowPtrA|csrRowPtrB|csrRowPtrC, and csrColIndA|csrColIndB|csrcolIndC respectively), and α and β are scalars. Since A and B have different sparsity patterns, cuSPARSE adopts a two-step approach to complete sparse matrix C. In the first step, the user allocates csrRowPtrC of m+1elements and uses function cusparseXcsrgeam2Nnz() to determine csrRowPtrC and the total number of nonzero elements. In the second step, the user gathers nnzC (number of nonzero elements of matrix C) from either (nnzC=*nnzTotalDevHostPtr) or (nnzC=csrRowPtrC(m)-csrRowPtrC(0)) and allocates csrValC, csrColIndC of nnzC elements respectively, then finally calls function cusparse[S|D|C|Z]csrgeam2() to complete matrix C.