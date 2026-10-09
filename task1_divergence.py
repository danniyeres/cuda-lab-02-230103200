import math
import time
import numpy as np
from numba import cuda


@cuda.jit
def kernel_uniform(d_arr):
    idx = cuda.grid(1)
    if idx < d_arr.size:
        val = d_arr[idx]
        for _ in range(1000):
            val = val * 1.0001 + 0.0001
        d_arr[idx] = val


@cuda.jit
def kernel_divergent(d_arr):
    idx = cuda.grid(1)
    if idx < d_arr.size:
        val = d_arr[idx]
        if idx % 2 == 0:
            for _ in range(1000):
                val = val * 1.0001 + 0.0001
        else:
            for _ in range(1000):
                val = (val - 0.0001) / 1.0001
        d_arr[idx] = val


@cuda.jit
def kernel_warp_aligned(d_arr):
    idx = cuda.grid(1)
    if idx < d_arr.size:
        val = d_arr[idx]
        warp_id = idx // 32
        if warp_id % 2 == 0:
            for _ in range(1000):
                val = val * 1.0001 + 0.0001
        else:
            for _ in range(1000):
                val = (val - 0.0001) / 1.0001
        d_arr[idx] = val


def benchmark_kernel(kernel, d_arr, threads_per_block, blocks_per_grid):
    kernel[blocks_per_grid, threads_per_block](d_arr)
    cuda.synchronize()

    start = time.perf_counter()
    for _ in range(10):
        kernel[blocks_per_grid, threads_per_block](d_arr)
        cuda.synchronize()
    end = time.perf_counter()

    return ((end - start) / 10.0) * 1000.0


def run_benchmark():
    N = 1_048_576
    threads_per_block = 256
    blocks_per_grid = math.ceil(N / threads_per_block)

    h_arr = np.ones(N, dtype=np.float32)
    d_arr = cuda.to_device(h_arr)

    time_a = benchmark_kernel(
        kernel_uniform, d_arr, threads_per_block, blocks_per_grid
    )
    time_b = benchmark_kernel(
        kernel_divergent, d_arr, threads_per_block, blocks_per_grid
    )
    time_c = benchmark_kernel(
        kernel_warp_aligned, d_arr, threads_per_block, blocks_per_grid
    )

    print(f"Kernel A (Uniform Path):        {time_a:.3f} ms")
    print(f"Kernel B (Full Divergence):     {time_b:.3f} ms")
    print(f"Kernel C (Warp-Aligned Branch): {time_c:.3f} ms")


if __name__ == "__main__":
    run_benchmark()