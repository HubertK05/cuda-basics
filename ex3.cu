// Including Libraries
#include <stdio.h>
#include <cuda_runtime.h>

// Defining the CUDA Kernel
__global__ void vectorAdd(int *a, int *out, int n) {
    int i = threadIdx.x % n;
    if (i < n) {
        out[i] = a[i] + 10;
    }
}


// Main Function
int main() {
    const int n = 4;
    int a[n], out[n];
    int *d_a, *d_out;

    // Allocate device memory
    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_out, n * sizeof(int));

    // Initialize arrays and copy to device
    for(int i = 0; i < n; i++) {
        a[i] = i;
    }
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    vectorAdd<<<1, n * 2>>>(d_a, d_out, n);

    // Copy result back to host
    cudaMemcpy(out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a); cudaFree(d_out);

    for (int i = 0; i < n; i++) {
        printf("%d + 10 = %d\n", a[i], out[i]);
    }

    return 0;
}
