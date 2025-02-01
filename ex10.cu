// Including Libraries
#include <stdio.h>
#include <cuda_runtime.h>

// Defining the CUDA Kernel
__global__ void vectorAdd(int *a, int *b, int *out, int n) {
    int i = threadIdx.x;
    __shared__ int shared[4];

    if (i < n) shared[i] = a[i] * b[i];

    for (int m = 2; m > 0; m >>= 1) {
        __syncthreads();
        if (i < m && i * (4 / m) + (2 / m) < n) shared[i * (4 / m)] = shared[i * (4 / m)] + shared[i * (4 / m) + (2 / m)];
    }

    if (i == 0) out[0] = shared[0];
}


// Main Function
int main() {
    const int n = 4;
    int a[n], b[n], out[1];
    int *d_a, *d_b, *d_out;

    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_b, n * sizeof(int));
    cudaMalloc((void**)&d_out, sizeof(int));

    for(int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i;
    }
    out[0] = 0;
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    vectorAdd<<<1, 4>>>(d_a, d_b, d_out, n);

    cudaMemcpy(out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);

    printf("dot product: %d\n", out[0]);

    return 0;
}
