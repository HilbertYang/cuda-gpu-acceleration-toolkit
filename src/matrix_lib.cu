#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define TILE_WIDTH 16

// ------------------------- CUDA CHECK -------------------------
#define CUDA_CHECK(call) do {                                  \
  cudaError_t e = (call);                                      \
  if (e != cudaSuccess) {                                      \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
            __FILE__, __LINE__, cudaGetErrorString(e));        \
    return;                                                    \
  }                                                           \
} while (0)

// ------------------------- TILED MATMUL KERNEL -------------------------
__global__ void matrixMultiplyTiled(const float *A, const float *B, float *C, int N) {
  __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float Pvalue = 0.0f;

  for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
    int aCol = m * TILE_WIDTH + tx;
    int bRow = m * TILE_WIDTH + ty;

    ds_A[ty][tx] = (Row < N && aCol < N) ? A[Row * N + aCol] : 0.0f;
    ds_B[ty][tx] = (bRow < N && Col < N) ? B[bRow * N + Col] : 0.0f;

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_WIDTH; ++k)
      Pvalue += ds_A[ty][k] * ds_B[k][tx];

    __syncthreads();
  }

  if (Row < N && Col < N)
    C[Row * N + Col] = Pvalue;
}

// ------------------------- EXPORTED: GPU MATMUL -------------------------
// Windows DLL export
extern "C" __declspec(dllexport)
void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N) {
  if (!h_A || !h_B || !h_C || N <= 0) return;

  size_t size = (size_t)N * (size_t)N * sizeof(float);
  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

  CUDA_CHECK(cudaMalloc((void**)&d_A, size));
  CUDA_CHECK(cudaMalloc((void**)&d_B, size));
  CUDA_CHECK(cudaMalloc((void**)&d_C, size));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH,
               (N + TILE_WIDTH - 1) / TILE_WIDTH);

  matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// =====================================================================
// Part 2: Convolution (CPU + CUDA) for grayscale uint32 images
// image: MxM, filter: NxN (N odd recommended)
// output: MxM
// border: zero-padding
// =====================================================================

// ------------------------- CPU CONV (reference) -------------------------
extern "C" __declspec(dllexport)
void cpu_convolve_u32(const uint32_t* img, int M,
                      const int32_t* filt, int N,
                      uint32_t* out) {
  if (!img || !filt || !out || M <= 0 || N <= 0) return;
  int r = N / 2;

  for (int y = 0; y < M; y++) {
    for (int x = 0; x < M; x++) {
      int64_t acc = 0;
      for (int fy = 0; fy < N; fy++) {
        for (int fx = 0; fx < N; fx++) {
          int iy = y + (fy - r);
          int ix = x + (fx - r);
          uint32_t pix = 0;
          if (iy >= 0 && iy < M && ix >= 0 && ix < M) {
            pix = img[iy * M + ix];
          }
          acc += (int64_t)pix * (int64_t)filt[fy * N + fx];
        }
      }
      if (acc < 0) acc = 0;
      if (acc > 255) acc = 255; // assume 8-bit grayscale range
      out[y * M + x] = (uint32_t)acc;
    }
  }
}

// ------------------------- CUDA CONV KERNEL -------------------------
__global__ void conv2d_u32_kernel(const uint32_t* img, int M,
                                  const int32_t* filt, int N,
                                  uint32_t* out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= M || y >= M) return;

  int r = N / 2;
  int64_t acc = 0;

  for (int fy = 0; fy < N; fy++) {
    for (int fx = 0; fx < N; fx++) {
      int iy = y + (fy - r);
      int ix = x + (fx - r);
      uint32_t pix = 0;
      if (iy >= 0 && iy < M && ix >= 0 && ix < M) {
        pix = img[iy * M + ix];
      }
      acc += (int64_t)pix * (int64_t)filt[fy * N + fx];
    }
  }

  if (acc < 0) acc = 0;
  if (acc > 255) acc = 255;
  out[y * M + x] = (uint32_t)acc;
}

// ------------------------- EXPORTED: GPU CONV -------------------------
extern "C" __declspec(dllexport)
void gpu_convolve_u32(const uint32_t* h_img, int M,
                      const int32_t* h_filt, int N,
                      uint32_t* h_out) {
  if (!h_img || !h_filt || !h_out || M <= 0 || N <= 0) return;

  size_t img_bytes  = (size_t)M * (size_t)M * sizeof(uint32_t);
  size_t filt_bytes = (size_t)N * (size_t)N * sizeof(int32_t);

  uint32_t *d_img = nullptr, *d_out = nullptr;
  int32_t *d_filt = nullptr;

  CUDA_CHECK(cudaMalloc((void**)&d_img, img_bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_out, img_bytes));
  CUDA_CHECK(cudaMalloc((void**)&d_filt, filt_bytes));

  CUDA_CHECK(cudaMemcpy(d_img, h_img, img_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_filt, h_filt, filt_bytes, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  conv2d_u32_kernel<<<grid, block>>>(d_img, M, d_filt, N, d_out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_out, d_out, img_bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_img); cudaFree(d_out); cudaFree(d_filt);
}
