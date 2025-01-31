// Including Libraries
#include <stdio.h>
#include <cuda_runtime.h>

// Defining the CUDA Kernel
__global__ void vectorAdd(int *a, int *out, int n) {
    int i = threadIdx.x;
    __shared__ int shared[8];

    if (i < 8) {
        for (int j = max(0, i - 2); j <= i; j++) {
            shared[i] += a[j];
        }
    }
    __syncthreads();
    out[i] = shared[i];
}


// Main Function
int main() {
    const int n = 8;
    int a[n], out[n];
    int *d_a, *d_out;

    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_out, n * sizeof(int));

    for(int i = 0; i < n; i++) {
        a[i] = i;
    }
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);

    vectorAdd<<<1, 8>>>(d_a, d_out, n);

    cudaMemcpy(out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_out);

    for (int i = 0; i < n; i++) {
        printf("sum of last three: %d\n", out[i]);
    }

    return 0;
}
