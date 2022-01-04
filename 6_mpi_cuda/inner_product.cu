#include <iostream> 
#include <mpi.h>


// and  make COMPILER=MPI_NVCC_
// get the MPI compile (mpicxx -showme:compile |sed 's/-pthread//g') and link options (mpicxx -showme:link |sed 's/-pthread//g')
// compile:  > nvcc -c -O3  -arch sm_30  -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi *.cu [*.cpp]
// link: > nvcc -lm  -lcudart -lcublas  -L/usr/lib/openmpi/lib -lmpi_cxx -lmpi -lopen-rte -lopen-pal -ldl -lnsl -lutil -lm -ld *.o -o main.MPI_NVCC_
// run code on 4 MPI processes: > mpirun -np 4 main.MPI_NVCC_
// or via make NVCC_MPI_

// define MPI compilers and options:
// > export OMPI_CXX= nvcc
// > export OMPI_CXXFLAGS= -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi     (= mpicxx -showme:compile |sed 's/-pthread//g')
// > export OMPI_LDFLAGS= -L/usr/lib/openmpi/lib -lmpi_cxx -lmpi -lopen-rte -lopen-pal -ldl -lnsl -lutil -lm -ld   ( = mpicxx -showme:link |sed 's/-pthread//g')
// > export OMPI_LIBS=                     (otherwise the nvcc linker cannot handle the option -Wl,--export-dynamic)
// compile:  > mpicxx -c -O3  -arch sm_30 *.cu [*.cpp]
// link: > mpicxx -lm  -lcudart -lcublas *.o -o main.NVCC_MPI_
// run code on 4 MPI processes: > mpirun -np 4 main.NVCC_MPI_

// kernel function on device  
__global__ void scalar(double *sg, const double *x, const double *y, int N)
{
	extern __shared__ double sdata[];
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int str = gridDim.x*blockDim.x;

	double s = 0.0;
	for (int i = idx; i < N; i += str)
		s+= x[i]*y[i];
	sdata[threadIdx.x] = s;

	__syncthreads();
	for (int s=blockDim.x >> 1; s>0; s>>=1) {
		if (threadIdx.x < s) {
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0)  sg[blockIdx.x] = sdata[0];
}


//    kernel function for addition on  one  block on device  
__global__ void add_1_block(double *s, int N)
{
	if (N>blockDim.x) return;
	extern __shared__ double sdata[];
	const int tid = threadIdx.x;
	if (tid<N) sdata[tid]=s[tid];
	else       sdata[tid]=0.0;

	__syncthreads();
	for (int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0)  s[0] = sdata[0];
}

//-------------------------------------------------------------
// Host function. Inner product calculated with device vectors
double dscapr_GPU(const int N, const double x_d[], const double y_d[])
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties (&prop, 0);

	const int blocksize = 4 * 64,
	      gridsize = prop.multiProcessorCount,
	      sBytes   = blocksize * sizeof(double);
	dim3 dimBlock(blocksize);
	dim3 dimGrid(gridsize);

	double sg, *s_d;  // device vector storing the partial sums
	cudaMalloc((void **) &s_d, blocksize * sizeof(double)); // temp. memory on device

	// call the kernel function with  dimGrid.x * dimBlock.x threads
	scalar <<< dimGrid, dimBlock, sBytes>>>(s_d, x_d, y_d, N);
	// power of 2 as number of treads per block
	const unsigned int oneBlock = 2 << (int)ceil(log(dimGrid.x + 0.0) / log(2.0));
	add_1_block <<< 1, oneBlock, oneBlock *sizeof(double)>>>(s_d, dimGrid.x);
	// copy data:  device --> host
	cudaMemcpy(&sg, s_d, sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(s_d);
	return sg;
}

//-------------------------------------------------------------
//  main function on host
int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	double *x_h, *y_h; // host data
	double *x_d, *y_d; // device data
	double sum, tstart, tgpu;
	const int N = 14000000,
	      nBytes = N*sizeof(double);
	const int LOOP = 100;

	// std::cout << gridsize << " x " << blocksize << " Threads\n";

	x_h = new double [N];
	y_h = new double [N];
	// allocate memory on device
	cudaMalloc((void **) &x_d, nBytes);
	cudaMalloc((void **) &y_d, nBytes);

	for (int i=0; i<N; i++)  { 
		x_h[i] = (i % 137)+1; y_h[i] = 1.0/x_h[i];
	}

	// copy data:  host --> device
	cudaMemcpy(x_d, x_h, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y_h, nBytes, cudaMemcpyHostToDevice);

	for (int k=0; k<LOOP; ++k) 
	{
		double loc_sum = dscapr_GPU(N, x_d, y_d);
		MPI_Allreduce(&loc_sum,&sum,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	}

	delete [] x_h; delete [] y_h; 
	cudaFree(x_d); cudaFree(y_d);
	MPI_Finalize();
	return 0;
}
