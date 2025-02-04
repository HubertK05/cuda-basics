// Including Libraries
#include <stdio.h>
#include <cuda_runtime.h>

// Defining the CUDA Kernel
__global__ void convolution(int *a, int *b, int *out, int n, int m) {
    int block_size = blockDim.x;
    int i = threadIdx.x;
    for (int j = 0; j <= m; j++) {
        for (int k = i; k < n; k += block_size) {
            if (k >= j) out[k - j] += a[k] * b[j];
        }
        __syncthreads();
    }
}


// Main Function
int main() {
    const int n = 15;
    const int m = 4;
    int a[n], b[m], out[n];
    int *d_a, *d_b, *d_out;

    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_b, m * sizeof(int));
    cudaMalloc((void**)&d_out, n * sizeof(int));

    for(int i = 0; i < n; i++) {
        a[i] = i;
        out[i] = 0;
    }
    for (int i = 0; i < m; i++) b[i] = i;
    
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, m * sizeof(int), cudaMemcpyHostToDevice);

    convolution<<<1, 8>>>(d_a, d_b, d_out, n, m);

    cudaMemcpy(out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);

    for (int i = 0; i < n; i++) {
        printf("%d ", out[i]);
    }
    printf("\n");

    return 0;
}
