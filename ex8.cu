// Including Libraries
#include <stdio.h>
#include <cuda_runtime.h>

// Defining the CUDA Kernel
__global__ void vectorAdd(int *a, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x;
    
    __shared__ int shared[4];
    if (local_i < 4) {
        shared[i] = a[i] + 10;
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

    vectorAdd<<<2, 4>>>(d_a, d_out, n);

    cudaMemcpy(out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_out);

    for (int i = 0; i < n; i++) {
        printf("%d + 10 = %d\n", a[i], out[i]);
    }

    return 0;
}
