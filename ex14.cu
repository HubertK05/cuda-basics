// Including Libraries
#include <stdio.h>
#include <cuda_runtime.h>

// Defining the CUDA Kernel
__global__ void matrix_mul(int **a, int **b, int **out, int n) {
    int i = threadIdx.x;
    int num_threads = blockDim.x;
    __shared__ int shared[8];

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            for (int j = i; j < n; j += num_threads) shared[j] = a[row][j] * b[j][col];
            for (int m = 4; m > 0; m >>= 1) {
                int step = 8 / m;
                int half_step = step / 2;
                __syncthreads();

                for (int j = i; j < m && j * step + half_step < n; j += num_threads) {
                    shared[j * step] = shared[j * step] + shared[j * step + half_step];
                }
            }
            __syncthreads();
            out[row][col] = shared[0];
        }
    }

    __syncthreads();
}


// Main Function
int main() {
    const int n = 8;
    int a[n][n], b[n][n], out[n][n];
    int **d_a, **d_b, **d_out, *a_rows[n], *b_rows[n], *out_rows[n];

    cudaMalloc((void**)&d_a, n * sizeof(int*));
    cudaMalloc((void**)&d_b, n * sizeof(int*));
    cudaMalloc((void**)&d_out, n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        cudaMalloc((void**)&a_rows[i], n * sizeof(int));
        cudaMalloc((void**)&b_rows[i], n * sizeof(int));
        cudaMalloc((void**)&out_rows[i], n * sizeof(int));
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = i * n + j;
            b[j][i] = i * n + j;
            out[i][j] = 0;
        }
    }

    cudaMemcpy(d_a, a_rows, n * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_rows, n * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out_rows, n * sizeof(int*), cudaMemcpyHostToDevice);
    for (int i = 0; i < n; i++) {
        cudaMemcpy(a_rows[i], a[i], n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(b_rows[i], b[i], n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(out_rows[i], out[i], n * sizeof(int), cudaMemcpyHostToDevice);
    }

    matrix_mul<<<1, 5>>>(d_a, d_b, d_out, n);

    for (int i = 0; i < n; i++) {
        cudaMemcpy(out[i], out_rows[i], n * sizeof(int), cudaMemcpyDeviceToHost);
    }

    for (int i = 0; i < n; i++) {
        cudaFree(a_rows[i]);
        cudaFree(b_rows[i]);
        cudaFree(out_rows[i]);
    }
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) printf("%d ", out[i][j]);
        printf("\n");
    }

    return 0;
}
