// Including Libraries
#include <stdio.h>
#include <cuda_runtime.h>

// Defining the CUDA Kernel
__global__ void array_sum(int **a, int *out, int rows, int cols) {
    int i = threadIdx.x;
    int num_threads = blockDim.x;
    __shared__ int shared[6];

    for (int col = 0; col < cols; col++) {
        for (int j = i; j < rows; j += num_threads) shared[j] = a[j][col];
        for (int m = 4; m > 0; m >>= 1) {
            int step = 8 / m;
            int half_step = step / 2;
            __syncthreads();

            for (int j = i; j < m && j * step + half_step < rows; j += num_threads) {
                shared[j * step] = shared[j * step] + shared[j * step + half_step];
            }
        }
        __syncthreads();
        out[col] = shared[0];
    }

    __syncthreads();
}


// Main Function
int main() {
    const int rows = 6;
    const int cols = 4;
    int a[rows][cols], out[cols];
    int **d_a, *a_rows[rows], *d_out;

    cudaMalloc((void**)&d_a, rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        cudaMalloc((void**)&a_rows[i], cols * sizeof(int));
    }
    cudaMalloc((void**)&d_out, cols * sizeof(int));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) a[i][j] = i * cols + j;
    }
    for (int i = 0; i < cols; i++) out[i] = 0;

    cudaMemcpy(d_a, a_rows, rows * sizeof(int*), cudaMemcpyHostToDevice);
    for (int i = 0; i < rows; i++) {
        cudaMemcpy(a_rows[i], a[i], cols * sizeof(int), cudaMemcpyHostToDevice);
    }

    array_sum<<<1, 4>>>(d_a, d_out, rows, cols);

    cudaMemcpy(out, d_out, cols * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows; i++) cudaFree(a_rows[i]);
    cudaFree(d_a); cudaFree(d_out);

    for (int i = 0; i < cols; i++) printf("sum of col %d: %d\n", i, out[i]);

    return 0;
}
