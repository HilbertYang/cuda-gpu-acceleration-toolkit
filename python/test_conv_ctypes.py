import ctypes
import numpy as np
import time
from pathlib import Path

# Optional: if you want real images
from PIL import Image

dll_path = Path(__file__).resolve().parent.parent / "lib" / "matrix.dll"
lib = ctypes.CDLL(str(dll_path))

# ---- ctypes signatures ----
u32_1d = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS")
i32_1d = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")

lib.cpu_convolve_u32.argtypes = [u32_1d, ctypes.c_int, i32_1d, ctypes.c_int, u32_1d]
lib.cpu_convolve_u32.restype = None

lib.gpu_convolve_u32.argtypes = [u32_1d, ctypes.c_int, i32_1d, ctypes.c_int, u32_1d]
lib.gpu_convolve_u32.restype = None


def load_grayscale_u8(path: Path, M: int) -> np.ndarray:
    """Load image, convert to grayscale, resize to MxM, return uint32 array (0..255)."""
    img = Image.open(path).convert("L").resize((M, M))
    arr = np.array(img, dtype=np.uint8)
    return arr.astype(np.uint32)


def save_grayscale_u8(arr_u32: np.ndarray, path: Path):
    """Save uint32 (0..255) MxM as PNG."""
    arr_u8 = np.clip(arr_u32, 0, 255).astype(np.uint8)
    Image.fromarray(arr_u8, mode="L").save(path)


def run_one(img_u32: np.ndarray, filt: np.ndarray, name: str):
    M = img_u32.shape[0]
    N = filt.shape[0]

    img_flat = np.ascontiguousarray(img_u32.ravel(), dtype=np.uint32)
    filt_flat = np.ascontiguousarray(filt.ravel(), dtype=np.int32)
    out_cpu = np.zeros((M*M,), dtype=np.uint32)
    out_gpu = np.zeros((M*M,), dtype=np.uint32)

    # CPU timing
    t0 = time.time()
    lib.cpu_convolve_u32(img_flat, M, filt_flat, N, out_cpu)
    t1 = time.time()

    # GPU timing
    t2 = time.time()
    lib.gpu_convolve_u32(img_flat, M, filt_flat, N, out_gpu)
    t3 = time.time()

    cpu_s = t1 - t0
    gpu_s = t3 - t2

    print(f"[{name}] M={M}, N={N}  CPU={cpu_s:.4f}s  GPU={gpu_s:.4f}s  speedup={cpu_s/gpu_s if gpu_s>0 else 0:.2f}x")
    return out_cpu.reshape(M, M), out_gpu.reshape(M, M), cpu_s, gpu_s


def main():
    out_dir = Path(__file__).resolve().parent / "conv_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick 2-3 sample images you have (put them in python/images/)
    img_dir = Path(__file__).resolve().parent / "images"
    images = [
        img_dir / "img1.png",
        img_dir / "img2.png",
        img_dir / "img3.png",
    ]

    # Filters (edge detection examples)
    # N = 3 and 5 and 7 (3 different Ns)
    sobel_x_3 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], dtype=np.int32)

    laplacian_3 = np.array([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0],
    ], dtype=np.int32)

    # Simple 5x5 box blur (not edge but good for variety)
    box_5 = np.ones((5, 5), dtype=np.int32)

    filters = [
        ("sobelx3", sobel_x_3),
        ("lap3", laplacian_3),
        ("box5", box_5),
    ]

    # 3 different Ms
    Ms = [256, 512, 1024]

    for img_path in images:
        if not img_path.exists():
            print(f"Missing image: {img_path}  (Put PNGs in python/images/ and name them img1.png/img2.png/img3.png)")
            continue

        for M in Ms:
            img_u32 = load_grayscale_u8(img_path, M)

            for fname, filt in filters:
                out_cpu, out_gpu, cpu_s, gpu_s = run_one(img_u32, filt, f"{img_path.stem}_{fname}")

                save_grayscale_u8(out_cpu, out_dir / f"{img_path.stem}_M{M}_{fname}_cpu.png")
                save_grayscale_u8(out_gpu, out_dir / f"{img_path.stem}_M{M}_{fname}_gpu.png")

    print(f"Saved outputs to: {out_dir}")

if __name__ == "__main__":
    main()
