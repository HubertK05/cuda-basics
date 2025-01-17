// Including Libraries
#include <stdio.h>
#include <cuda_runtime.h>

// Defining the CUDA Kernel
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}


// Main Function
int main() {
    const int n = 256;
    int a[n], b[n], c[n];
    int *d_a, *d_b, *d_c;

    // Allocate device memory
    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_b, n * sizeof(int));
    cudaMalloc((void**)&d_c, n * sizeof(int));

    // Initialize arrays and copy to device
    for(int i = 0; i < n; i++) {
        a[i] = b[i] = i;
    }
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    vectorAdd<<<1, n>>>(d_a, d_b, d_c, n);

    // Copy result back to host
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    for (int i = 0; i < n; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    return 0;
}
