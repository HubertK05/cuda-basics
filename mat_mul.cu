// Including Libraries
#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_AMOUNT_X 1
#define BLOCK_AMOUNT_Y 1
#define BLOCK_SIZE 8

// Defining the CUDA Kernel
__global__ void matrix_mul(int **a, int **b, int **out, int n) {
    int j = threadIdx.x;
    int i = threadIdx.y;
    const int thread_len = blockDim.x;
    int block_j = blockIdx.x;
    int block_i = blockIdx.y;
    __shared__ int shared_a[BLOCK_SIZE][512];
    __shared__ int shared_b[512][BLOCK_SIZE];
    __shared__ int shared_out[BLOCK_SIZE][BLOCK_SIZE];


    for (int subm_i = block_i; subm_i < n / thread_len; subm_i += BLOCK_AMOUNT_X) {
        for (int row = i; row < thread_len; row += thread_len) {
            for (int col = j; col < n; col += thread_len) {
                shared_a[row][col] = a[row + thread_len * subm_i][col];
            }
        }
        for (int subm_j = block_j; subm_j < n / thread_len; subm_j += BLOCK_AMOUNT_Y) {
            for (int row = i; row < n; row += thread_len) {
                for (int col = j; col < thread_len; col += thread_len) {
                    shared_b[row][col] = b[row][col + thread_len * subm_j];
                }
            }
            
            for (int row = i; row < thread_len; row += thread_len) {
                for (int col = j; col < thread_len; col += thread_len) {
                    shared_out[row][col] = 0;
                }
            }

            for (int row = i; row < thread_len; row += thread_len) {
                for (int col = j; col < thread_len; col += thread_len) {
                    for (int k = 0; k < n; k++) {
                        shared_out[row][col] += shared_a[row][k] * shared_b[k][col];
                    }
                }
            }

            for (int row = i; row < thread_len; row += thread_len) {
                for (int col = j; col < thread_len; col += thread_len) {
                    out[row + thread_len * subm_i][col + thread_len * subm_j] = shared_out[row][col];
                }
            }

            __syncthreads();
        }
    }
}


// Main Function
int main() {
    const int n = 512;
    
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

    dim3 blocks(BLOCK_AMOUNT_X, BLOCK_AMOUNT_Y);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    matrix_mul<<<blocks, threads>>>(d_a, d_b, d_out, n);

    for (int i = 0; i < n; i++) {
        cudaMemcpy(out[i], out_rows[i], n * sizeof(int), cudaMemcpyDeviceToHost);
    }

    for (int i = 0; i < n; i++) {
        cudaFree(a_rows[i]);
        cudaFree(b_rows[i]);
        cudaFree(out_rows[i]);
    }
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);

    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) printf("%d ", out[i][j]);
    //     printf("\n");
    // }

    for (int i = 0; i < n; i++) printf("%d ", out[0][i]);
    printf("\n");

    return 0;
}
