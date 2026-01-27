# GPU-Accelerated Matrix Multiplication & Python Integration

## Overview

This project implements a high-performance matrix multiplication framework optimized for both CPU and GPU execution. It provides multiple CUDA implementations, performance benchmarking tools, and a Python interface powered by a custom shared library.

The system is designed to explore scalable numerical computing, GPU optimization strategies, and efficient cross-language acceleration between C/CUDA and Python.

---

## Key Features

### Multi-Backend Matrix Multiplication
- CPU baseline implementation (C)
- Naïve CUDA GPU kernel
- Optimized tiled CUDA kernel using shared memory
- cuBLAS SGEMM implementation for maximum performance

### Performance Benchmarking
- Runtime benchmarking across multiple matrix sizes
- CPU vs GPU speedup analysis
- Performance scaling evaluation for large workloads

### Python GPU Acceleration
- CUDA kernels compiled into a shared library (`.so`)
- Python bindings using `ctypes`
- GPU-accelerated matrix multiplication callable directly from Python

### Extensible CUDA Framework
- Modular kernel design
- Ready for extension to convolution, image processing, and ML workloads

---


## Directory Structure

lab-matmul/
├── bin/ # Compiled executables
├── doc/ # Documentation and reports
├── include/ # Header files
├── lib/ # Shared libraries (.so)
├── python/ # Python interface and scripts
├── results/ # Benchmark results and performance logs
├── scripts/ # Build & automation scripts
├── src/ # C / CUDA source code
└── readme.md # Project documentation

---

## Technologies Used

- C / C++
- CUDA
- NVIDIA cuBLAS
- Python
- ctypes
- NVIDIA GPU Toolkit