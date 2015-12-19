#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

/* Program Parameters */
#define MAXN 8000  /* Max value of N */
int N;  /* Matrix size */

// Thread block size
#define BLOCK_SIZE 16

/* Matrices */
float A[MAXN][MAXN], B[MAXN][MAXN];

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
/* ------------------ Cuda Code --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][] and B[][],
 * defined in the beginning of this code.  B[][] is initialized to zeros.
 */

__global__ void matrixMean(float* d_in, float* d_mean, int N)
{
  extern __shared__ float sdata[];

  //each thread loads one element from global to shared mem
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  
  unsigned int tid = threadIdx.y;
  unsigned int i = idx_y * N + idx_x;
  sdata[tid] = d_in[i];
  __syncthreads();

  // do reduction in shared mem
  for(unsigned int s=1; s < blockDim.y; s *= 2)
  {
    if(tid +s < N)
    {
      if(tid % (2*s) == 0)
      {
        sdata[tid] += sdata[tid + s];
      }
    }
  __syncthreads();
  }

  // write result for this block to global mem
  if(tid == 0)
  {
    d_mean[blockIdx.x] = sdata[0]/(float) N;
  }
}

__global__ void matrixSD(float* d_in, float* d_mean, float* d_sd, int N)
{
  extern __shared__ float sdata1[];

  //each thread loads one element from global to shared mem
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  
  unsigned int tid = threadIdx.y;
  unsigned int i = idx_y * N + idx_x;
  sdata1[tid] = powf(d_in[i] - d_mean[blockIdx.x], 2.0);
  __syncthreads();

  // do reduction in shared mem
  for(unsigned int s=1; s < blockDim.y; s *= 2)
  {
    if(tid +s < N)
    {
      if(tid % (2*s) == 0)
      {
        sdata1[tid] += sdata1[tid + s];
      }
    }
  __syncthreads();
  }

  // write result for this block to global mem
  if(tid == 0)
    d_sd[blockIdx.x] = sqrtf(sdata1[0]/(float) N);
}

__global__ void matrixNorm(float* d_in, float* d_out, float* d_mean, float* d_sd, int N)
{
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  
  unsigned int i = idx_y * N + idx_x;

  if (d_sd[blockIdx.y] == 0.0)
    d_out[i] = 0.0;
  else
    d_out[i] = (d_in[i] - d_mean[blockIdx.x]) / d_sd[blockIdx.x];
}

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int seed = 0;  /* Random seed */
  char uid[32]; /*User name */

  /* Read command-line arguments */
  srand(time_seed());  /* Randomize */

  if (argc == 3) {
    seed = atoi(argv[2]);
    srand(seed);
    printf("Random seed = %i\n", seed);
  } 
  if (argc >= 2) {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
  }
  else {
    printf("Usage: %s <matrix_dimension> [random seed]\n",
           argv[0]);    
    exit(0);
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B*/
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = row * N + col;
      //A[row][col] = (float)rand() / 32768.0;
      B[row][col] = 0.0;
    }
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
      printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
  }
}

void print_B() {
    int row, col;

    if (N < 10) {
        printf("\nB =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%1.10f%s", B[row][col], (col < N-1) ? ", " : ";\n\t");
            }
        }
    }
}

int main(int argc, char **argv) {
  /* Timing variables */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  /* Process program parameters */
  parameters(argc, argv);

  /* Initialize A and B */
  //initialize_inputs();

  /* Print input matrices */
  //print_inputs();

  /* Gaussian Elimination */
  float* A1 = new float [N * N];
  float* B1 = new float [N * N];

  int i,j;
  printf("\nInitializing...\n");

  for(i=0;i<N;i++)
  {
    for(j=0;j<N;j++)
    {
      A1[j* N + i] = (float)rand() / 32768.0;
    }
  }
  if (N < 10) {
    printf("\nA1 =\n\t");
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
      printf("%5.2f%s", A1[i* N + j], (j < N-1) ? ", " : ";\n\t");
      }
    }
  }

  float* d_in;
  float* d_out;
  float* d_mean;
  float* d_sd;
  size_t size2d = N * N * sizeof(float);
  size_t size1d = N * sizeof(float);

  //allocated the device memory for source array
  cudaMalloc(&d_in, size2d);
  cudaMemcpy(d_in, A1, size2d, cudaMemcpyHostToDevice);

  //allocate the device memory for destination array
  cudaMalloc(&d_out, size2d);

  //allocate the device memory for mean arry
  cudaMalloc(&d_mean, size1d);

  //allocate the device memory for sd array
  cudaMalloc(&d_sd, size1d);

  //dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  //dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);

  dim3 dimBlock;
  dim3 dimGrid;

  if( N < 1024)
  {
    dimBlock.x = 1;
    dimBlock.y = N;
    dimGrid.x = N;
    dimGrid.y = 1;
  }
  else
  {
    dimBlock.x = 1;
    dimBlock.y = 1024;
    dimGrid.x = N;
    dimGrid.y = 1;
  }

  /* Start Clock */
  printf("\nStarting clock.\n");
  cudaEventRecord(start);
  gettimeofday(&etstart, &tzdummy);
  etstart2 = times(&cputstart);

  matrixMean<<<dimGrid, dimBlock, size1d>>>(d_in, d_mean, N);
  cudaDeviceSynchronize();
  matrixSD<<<dimGrid, dimBlock, size1d>>>(d_in, d_mean, d_sd, N);
  cudaDeviceSynchronize();
  matrixNorm<<<dimGrid, dimBlock>>>(d_in, d_out, d_mean, d_sd, N);
  cudaDeviceSynchronize();

  /* Stop Clock */
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");

  cudaMemcpy(B1, d_out, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  if (N < 10) {
  printf("\nB1 =\n\t");
    for (i= 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%1.10f%s", B1[i* N + j], (j < N-1) ? ", " : ";\n\t");
        }
    }
  }

  /* Display timing results */
  printf("\nElapsed time CPU Time = %g ms.\n", (float)(usecstop - usecstart)/(float)1000);
  printf("Elapsed time Cuda Time = %g ms \n",milliseconds);
  printf("Effective Bandwidth (GB/s): %f \n", (2*size2d/milliseconds)/1e6);
  float mean = N * log2((float)N) + N;
  float sd = N * log2((float)N) + (2*N) + (2*N*N);
  float norm = 2 * N * N;
  printf("Effective Throughput (GFLOPS/s): %f \n", ((mean+sd+norm)*1e-9)/(milliseconds*1e-3)); 
  printf("--------------------------------------------\n");

  //deallocate device memory
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_mean);
  cudaFree(d_sd);

  free(A1);
  free(B1);

  exit(0);
}