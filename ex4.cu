// Including Libraries
#include <stdio.h>
#include <cuda_runtime.h>

// Defining the CUDA Kernel
__global__ void vectorAdd(int **a, int **out, int n) {
    int i = threadIdx.x;
    printf("%d ", a[0][i]);
    if (i < n) {
        out[0][i] = a[0][i] + 10;
    }
}


// Main Function
int main() {
    const int n = 4;
    int a[n][n], out[n][n];
    int **d_a, **d_out, *d_a_rows[n], *d_out_rows[n];

    // Allocate device memory
    printf("Doing main arrays\n");
    cudaMalloc((void**)&d_a, n * sizeof(int *));
    cudaMalloc((void**)&d_out, n * sizeof(int *));
    printf("Main arrays done\n");
    for (int i = 0; i < n; i++) {
        printf("Subarray %d\n", i);
        cudaMalloc((void**)&d_a_rows[i], n * sizeof(int));
        cudaMalloc((void**)&d_out_rows[i], n * sizeof(int));
    }

    // Initialize arrays and copy to device
    printf("Initializing arrays\n");
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            a[i][j] = i * n + j;
        }
    }
    printf("Copying memory\n");
    cudaMemcpy(d_a, d_a_rows, n * sizeof(int *), cudaMemcpyHostToDevice);

    // Launch kernel
    printf("Running kernel\n");
    vectorAdd<<<1, (n + 1) * (n + 1)>>>(d_a, d_out, n);

    // Copy result back to host
    printf("\nCopying results\n");
    cudaMemcpy(d_out_rows, d_out, n * sizeof(int *), cudaMemcpyDeviceToHost);

    printf("Cleaning up\n");
    // Cleanup
    for (int i = 0; i < n; i++) {
        cudaFree(d_a_rows[i]);
        cudaFree(d_out_rows[i]);
    }
    cudaFree(d_a); cudaFree(d_out);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d->%d ", a[i][j], out[i][j]);
        }
        printf("\n");
    }

    return 0;
}
