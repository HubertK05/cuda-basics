// Including Libraries
#include <stdio.h>
#include <cuda_runtime.h>

// Defining the CUDA Kernel
__global__ void vectorAdd(int *a, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + 10;
    }
}


// Main Function
int main() {
    const int n = 9;
    int a[n], out[n];
    int *d_a, *d_out;

    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_out, n * sizeof(int));

    for(int i = 0; i < n; i++) {
        a[i] = i;
    }
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);

    vectorAdd<<<3, 4>>>(d_a, d_out, n);

    cudaMemcpy(out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_out);

    for (int i = 0; i < n; i++) {
        printf("%d + 10 = %d\n", a[i], out[i]);
    }

    return 0;
}
