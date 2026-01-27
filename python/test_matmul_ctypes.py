import ctypes
import numpy as np
import time
from pathlib import Path

dll_path = Path(__file__).resolve().parent.parent / "lib" / "matrix.dll"
lib = ctypes.CDLL(str(dll_path))

lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]
lib.gpu_matrix_multiply.restype = None

N = 1024
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

t0 = time.time()
lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N)
t1 = time.time()

print(f"Python call to GPU tiled matmul completed in {t1 - t0:.4f} seconds")
print("C[0,0] =", C[0,0])
