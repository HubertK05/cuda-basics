// Including Libraries
#include <stdio.h>
#include <cuda_runtime.h>

// Defining the CUDA Kernel
__global__ void vectorAdd(int **a, int **out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = i / n;
    if (x < n) {
        int y = i % n;
        out[x][y] = a[x][y] + 10;
    }
}


// Main Function
int main() {
    const int n = 5;

    int a[n][n], out[n][n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = i * n + j;
        }
    }

    int **a_ptr, **out_ptr;
    int *a_rows[n], *out_rows[n];
    cudaMalloc((void**)&a_ptr, n * sizeof(int*));
    cudaMalloc((void**)&out_ptr, n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        cudaMalloc((void**)&a_rows[i], n * sizeof(int));
        cudaMalloc((void**)&out_rows[i], n * sizeof(int));
    }

    cudaMemcpy(a_ptr, a_rows, n * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(out_ptr, out_rows, n * sizeof(int*), cudaMemcpyHostToDevice);
    for (int i = 0; i < n; i++) {
        cudaMemcpy(a_rows[i], a[i], n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(out_rows[i], out[i], n * sizeof(int), cudaMemcpyHostToDevice);
    }

    vectorAdd<<<4, 9>>>(a_ptr, out_ptr, n);

    for (int i = 0; i < n; i++) {
        cudaMemcpy(out[i], out_rows[i], n * sizeof(int), cudaMemcpyDeviceToHost);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d->%d ", a[i][j], out[i][j]);
        }
        printf("\n");
    }

    return 0;
}
