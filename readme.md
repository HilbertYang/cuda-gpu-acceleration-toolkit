# EE533 Matrix Multiplication Benchmark
## CPU vs CUDA vs Optimized CUDA vs cuBLAS

This project benchmarks large matrix multiplication implementations across CPU and GPU platforms.
The goal is to analyze performance differences, optimization effects, and scalability as matrix size grows.

---

## Learning Objectives

By the end of this lab, we are able to:

1. Write and execute a C program performing large matrix multiplication on a CPU.
2. Measure CPU execution performance for different matrix sizes.
3. Port the CPU implementation to CUDA and execute it on a GPU.
4. Deploy and run CUDA programs on a GPU-enabled virtual machine (e.g., Google Cloud).
5. Optimize CUDA kernels to improve GPU performance.
6. Compare performance among CPU, naïve CUDA, optimized CUDA, and cuBLAS implementations.
7. Analyze performance scaling behavior as the problem size increases.
8. Create CUDA-based shared libraries and utilize GPU acceleration from Python.

