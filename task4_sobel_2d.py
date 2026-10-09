import math
import numpy as np
from numba import cuda


@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)

    if row < rows and col < cols:
        if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
            d_out[row, col] = 0.0
        else:
            # 3x3 Sobel Horizontal operator gradient:
            # [-1  0  1]
            # [-2  0  2]
            # [-1  0  1]
            val = (
                -1.0 * d_in[row - 1, col - 1]
                + 1.0 * d_in[row - 1, col + 1]
                - 2.0 * d_in[row, col - 1]
                + 2.0 * d_in[row, col + 1]
                - 1.0 * d_in[row + 1, col - 1]
                + 1.0 * d_in[row + 1, col + 1]
            )
            d_out[row, col] = val


def run_sobel(h_img):
    rows, cols = h_img.shape
    d_in = cuda.to_device(h_img.astype(np.float32))
    d_out = cuda.device_array((rows, cols), dtype=np.float32)

    threads_2d = (16, 16)
    blocks_x = math.ceil(cols / threads_2d[0])
    blocks_y = math.ceil(rows / threads_2d[1])
    blocks_2d = (blocks_x, blocks_y)

    sobel_x_kernel[blocks_2d, threads_2d](d_in, d_out, rows, cols)
    cuda.synchronize()

    return d_out.copy_to_host()


if __name__ == "__main__":
    test_img = np.ones((64, 64), dtype=np.float32)
    sobel_res = run_sobel(test_img)

    assert np.max(np.abs(sobel_res[1:-1, 1:-1])) < 1e-5
    assert np.all(sobel_res[0, :] == 0.0)
    assert np.all(sobel_res[:, 0] == 0.0)
    print("TASK 4 PASSED: 2D Sobel Horizontal Convolution verified.")