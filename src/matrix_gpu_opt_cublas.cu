#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do {                                        \
  cudaError_t err__ = (call);                                         \
  if (err__ != cudaSuccess) {                                         \
    fprintf(stderr, "CUDA error %s:%d: %s (%d)\n",                    \
            __FILE__, __LINE__, cudaGetErrorName(err__), (int)err__); \
    fprintf(stderr, "  %s\n", cudaGetErrorString(err__));             \
    exit(1);                                                          \
  }                                                                   \
} while(0)

#define CUBLAS_CHECK(call) do {                                       \
  cublasStatus_t st__ = (call);                                       \
  if (st__ != CUBLAS_STATUS_SUCCESS) {                                \
    fprintf(stderr, "cuBLAS error %s:%d: status=%d\n",                \
            __FILE__, __LINE__, (int)st__);                           \
    exit(1);                                                          \
  }                                                                   \
} while(0)

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---------------------- Naive Kernel ----------------------
__global__ void matmul_naive(const float *A, const float *B, float *C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

// ---------------------- Tiled Kernel ----------------------
#ifndef TILE
#define TILE 16
#endif

__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C, int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < N; t += TILE) {
    int a_col = t + threadIdx.x;
    int b_row = t + threadIdx.y;

    As[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

    __syncthreads();
  }

  if (row < N && col < N) C[row * N + col] = sum;
}

// ---------------------- Timing helpers ----------------------
static float time_event_begin(cudaEvent_t* start, cudaEvent_t* stop) {
  CUDA_CHECK(cudaEventCreate(start));
  CUDA_CHECK(cudaEventCreate(stop));
  CUDA_CHECK(cudaEventRecord(*start));
  return 0.0f;
}
static float time_event_end(cudaEvent_t start, cudaEvent_t stop) {
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 1024;
  if (N <= 0) { fprintf(stderr, "Invalid N\n"); return 1; }

  // GPU info
  int devCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&devCount));
  printf("CUDA device count: %d\n", devCount);
  if (devCount == 0) return 1;
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Using GPU %d: %s, cc %d.%d\n", dev, prop.name, prop.major, prop.minor);

  size_t size = (size_t)N * (size_t)N * sizeof(float);

  float *h_A = (float*)malloc(size);
  float *h_B = (float*)malloc(size);
  float *h_C = (float*)malloc(size);
  if (!h_A || !h_B || !h_C) { fprintf(stderr, "malloc failed\n"); return 1; }

  srand(0);
  for (int i = 0; i < N * N; i++) {
    h_A[i] = (rand() % 100) / 100.0f;
    h_B[i] = (rand() % 100) / 100.0f;
  }

  float *d_A = NULL, *d_B = NULL, *d_C = NULL;
  CUDA_CHECK(cudaMalloc((void**)&d_A, size));
  CUDA_CHECK(cudaMalloc((void**)&d_B, size));
  CUDA_CHECK(cudaMalloc((void**)&d_C, size));
  CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

  // -------------------- Naive timing --------------------
  dim3 block_naive(16, 16);
  dim3 grid_naive(ceil_div(N, (int)block_naive.x), ceil_div(N, (int)block_naive.y));
  matmul_naive<<<grid_naive, block_naive>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t s1, e1;
  time_event_begin(&s1, &e1);
  matmul_naive<<<grid_naive, block_naive>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaGetLastError());
  float naive_ms = time_event_end(s1, e1);

  // -------------------- Tiled timing --------------------
  dim3 block_tiled(TILE, TILE);
  dim3 grid_tiled(ceil_div(N, TILE), ceil_div(N, TILE));
  matmul_tiled<<<grid_tiled, block_tiled>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t s2, e2;
  time_event_begin(&s2, &e2);
  matmul_tiled<<<grid_tiled, block_tiled>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaGetLastError());
  float tiled_ms = time_event_end(s2, e2);

  // -------------------- cuBLAS timing --------------------
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  // Warm-up cuBLAS
  CUBLAS_CHECK(cublasSgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           N, N, N,
                           &alpha,
                           d_B, N,   // note: swapped A/B per lab snippet
                           d_A, N,
                           &beta,
                           d_C, N));
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t s3, e3;
  time_event_begin(&s3, &e3);
  CUBLAS_CHECK(cublasSgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           N, N, N,
                           &alpha,
                           d_B, N,
                           d_A, N,
                           &beta,
                           d_C, N));
  float cublas_ms = time_event_end(s3, e3);

  CUBLAS_CHECK(cublasDestroy(handle));

  // Copy back once (optional)
  CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

  printf("\nN=%d\n", N);
  printf("Naive CUDA:   %.3f ms\n", naive_ms);
  printf("Tiled CUDA:   %.3f ms\n", tiled_ms);
  printf("cuBLAS SGEMM: %.3f ms\n", cublas_ms);
  if (tiled_ms > 0)  printf("Speedup (Naive/Tiled): %.2fx\n", naive_ms / tiled_ms);
  if (cublas_ms > 0) printf("Speedup (Tiled/cuBLAS): %.2fx (lower is closer)\n", tiled_ms / cublas_ms);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A); free(h_B); free(h_C);
  return 0;
}
