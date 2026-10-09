import numpy as np
from numba import cuda


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):
    N = len(h_arr)
    d_arr = cuda.to_device(h_arr.astype(np.float32))

    threads_per_block = 256
    blocks_per_grid = 64

    grid_stride_scale_kernel[blocks_per_grid, threads_per_block](
        d_arr, factor, N
    )
    cuda.synchronize()

    return d_arr.copy_to_host()


if __name__ == "__main__":
    N = 100_000
    test_arr = np.ones(N, dtype=np.float32)
    factor = 4.25

    res = run_grid_stride(test_arr, factor)
    assert np.allclose(res, factor)
    print("TASK 3 PASSED: All elements scaled uniformly.")