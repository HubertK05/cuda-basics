// Including Libraries
#include <stdio.h>
#include <cuda_runtime.h>

// Defining the CUDA Kernel
__global__ void vectorBroadcast(int *a, int *b, int **out, int n, int m) {
    int i = threadIdx.x / m;
    int j = threadIdx.x % m;
    if (i < n) {
        out[i][j] = a[j] + b[i];
    }
}


// Main Function
int main() {
    const int n = 4;
    const int m = 5;

    int a[m], b[n], out[n][m];
    
    for (int i = 0; i < m; i++) a[i] = i;
    for (int i = 0; i < n; i++) b[i] = 5 + i * 2;

    int *a_ptr, *b_ptr, **out_ptr;
    int *out_rows[n];
    cudaMalloc((void**)&a_ptr, m * sizeof(int));
    cudaMalloc((void**)&b_ptr, n * sizeof(int));
    cudaMalloc((void**)&out_ptr, n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        cudaMalloc((void**)&out_rows[i], m * sizeof(int));
    }

    cudaMemcpy(a_ptr, a, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_ptr, b, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_ptr, out_rows, n * sizeof(int*), cudaMemcpyHostToDevice);
    for (int i = 0; i < n; i++) {
        cudaMemcpy(out_rows[i], out[i], m * sizeof(int), cudaMemcpyHostToDevice);
    }

    vectorBroadcast<<<1, (n + 1) * (m + 1)>>>(a_ptr, b_ptr, out_ptr, n, m);

    for (int i = 0; i < n; i++) {
        cudaMemcpy(out[i], out_rows[i], m * sizeof(int), cudaMemcpyDeviceToHost);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("(%d, %d)->%d ", a[j], b[i], out[i][j]);
        }
        printf("\n");
    }

    return 0;
}
