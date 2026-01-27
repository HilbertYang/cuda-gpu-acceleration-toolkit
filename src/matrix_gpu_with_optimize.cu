#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define CUDA_CHECK(call) do {                                        \
  cudaError_t err__ = (call);                                         \
  if (err__ != cudaSuccess) {                                         \
    fprintf(stderr, "CUDA error %s:%d: %s (%d)\n",                    \
            __FILE__, __LINE__, cudaGetErrorName(err__), (int)err__); \
    fprintf(stderr, "  %s\n", cudaGetErrorString(err__));             \
    exit(1);                                                          \
  }                                                                   \
} while(0)

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---------------------- Naive Kernel ----------------------
__global__ void matrixMultiplyGPU_naive(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        // Global memory reads each iteration (naive)
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ---------------------- Optimized Kernel (Tiled Shared Memory) ----------------------
#ifndef TILE
#define TILE 16
#endif

__global__ void matrixMultiplyGPU_tiled(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // Iterate over tiles
    for (int t = 0; t < N; t += TILE) {
        // Load A tile
        int a_col = t + threadIdx.x;
        if (row < N && a_col < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile
        int b_row = t + threadIdx.y;
        if (b_row < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ---------------------- Timing Helper ----------------------
static float time_kernel(void (*launch)(dim3, dim3, const float*, const float*, float*, int),
                         dim3 grid, dim3 block,
                         const float* dA, const float* dB, float* dC, int N) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warm-up
    launch(grid, block, dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    launch(grid, block, dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

// Wrappers so we can pass "launchers" to time_kernel
static void launch_naive(dim3 grid, dim3 block, const float* dA, const float* dB, float* dC, int N) {
    matrixMultiplyGPU_naive<<<grid, block>>>(dA, dB, dC, N);
}
static void launch_tiled(dim3 grid, dim3 block, const float* dA, const float* dB, float* dC, int N) {
    // block must be (TILE, TILE)
    matrixMultiplyGPU_tiled<<<grid, block>>>(dA, dB, dC, N);
}

int main(int argc, char **argv) {
    // ---- GPU info ----
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    printf("CUDA device count: %d\n", devCount);
    if (devCount == 0) {
        fprintf(stderr, "No CUDA device found.\n");
        return 1;
    }

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Using GPU %d: %s, cc %d.%d\n", dev, prop.name, prop.major, prop.minor);

    // ---- Parse N ----
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    if (N <= 0) {
        fprintf(stderr, "Invalid N.\n");
        return 1;
    }
    size_t size = (size_t)N * (size_t)N * sizeof(float);

    // ---- Host alloc/init ----
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_C2 = (float *)malloc(size);
    if (!h_A || !h_B || !h_C || !h_C2) {
        fprintf(stderr, "Host malloc failed.\n");
        return 1;
    }

    srand(0);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (rand() % 100) / 100.0f;
        h_B[i] = (rand() % 100) / 100.0f;
    }

    // ---- Device alloc/copy ----
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // ---- Launch configs ----
    dim3 block_naive(16, 16);
    dim3 grid_naive(ceil_div(N, (int)block_naive.x), ceil_div(N, (int)block_naive.y));

    dim3 block_tiled(TILE, TILE);
    dim3 grid_tiled(ceil_div(N, TILE), ceil_div(N, TILE));

    // ---- Time Naive ----
    float naive_ms = time_kernel(launch_naive, grid_naive, block_naive, d_A, d_B, d_C, N);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // ---- Time Tiled ----
    float tiled_ms = time_kernel(launch_tiled, grid_tiled, block_tiled, d_A, d_B, d_C, N);
    CUDA_CHECK(cudaMemcpy(h_C2, d_C, size, cudaMemcpyDeviceToHost));

    // ---- (Optional) quick correctness check on a few entries ----
    // Not a full verification, just sanity.
    int checks = 5;
    int ok = 1;
    for (int i = 0; i < checks; i++) {
        int idx = (i * 997) % (N * N);
        float diff = h_C[idx] - h_C2[idx];
        if (diff < 0) diff = -diff;
        if (diff > 1e-2f) { ok = 0; break; }
    }

    printf("Naive CUDA execution time (N=%d): %f ms\n", N, naive_ms);
    printf("Tiled  CUDA execution time (N=%d): %f ms\n", N, tiled_ms);
    if (tiled_ms > 0.0f) {
        printf("Speedup (Naive / Tiled): %fx\n", naive_ms / tiled_ms);
    }
    printf("Sanity check (naive vs tiled): %s\n", ok ? "PASS" : "FAIL (check implementation)");

    // ---- Cleanup ----
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C2);

    return 0;
}
