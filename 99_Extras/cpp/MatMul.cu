#include <stdio.h>
#include <stdlib.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void matmul_cpu(float* a, float *b, float* c, int N){
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            for(int k = 0; k < N; ++k){
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}

__global__ void matmul_gpu(float* a, float *b, float* c, int N){

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < N && COL < N) {
        float tmpSum = 0.0f;
        for (int i = 0; i < N; i++) {
            tmpSum += a[ROW * N + i] * b[i * N + COL];
            printf("a: %f\n", a[ROW * N + i]);
            printf("b: %f\n", b[i * N + COL]);
            printf("\n");
        }
        c[ROW * N + COL] = tmpSum;
        printf("(%d, %d): %f\n", ROW, COL, tmpSum);
    }
}

void random_init(float *a, int N){

    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            a[i * N + j] = rand() % 100;
        }
    }
}

void zero_init(float *a, int N){

    for(int i = 0; i < N; ++i){
        for(int j = 0; j < N; ++j){
            a[i * N + j] = 0;
        }
    }
}

void print_matrix(float *a, int N){

    for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
            printf("%f ", a[i*N+j]);
        }
        printf("\n");
    }
}

int main(){

    float *a;
    float *b;
    float *c;
    float *h_verify;

    int N = 2;
    int num_bytes = N * N * sizeof(float);
    a = (float*)malloc(num_bytes);
    b = (float*)malloc(num_bytes);
    c = (float*)malloc(num_bytes);
    h_verify = (float*)malloc(num_bytes);

    // device vectors
    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc((void**)&d_a, num_bytes);
    cudaMalloc((void**)&d_b, num_bytes);
    cudaMalloc((void**)&d_c, num_bytes);

    // Initialize
    random_init(a, N);
    random_init(b, N);
    zero_init(c, N);

    // Copy matrices to device
    cudaMemcpy(d_a, a, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, num_bytes, cudaMemcpyHostToDevice);

    // matmul
    matmul_cpu(a, b, c, N);

    // GPU matmul
    dim3 blocksPerGrid(16, 16, 1);
    dim3 threadsPerBlock(16, 16, 1);
    matmul_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    gpuErrchk(cudaMemcpy(h_verify, d_c, num_bytes, cudaMemcpyDeviceToHost));

    // Print matrix
    print_matrix(a, N);
    print_matrix(b, N);
    print_matrix(c, N);
    print_matrix(h_verify, N);

    // Free matrices
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaDeviceSynchronize();

    return 0;
}