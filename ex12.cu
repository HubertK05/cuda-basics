// Including Libraries
#include <stdio.h>
#include <cuda_runtime.h>

// Defining the CUDA Kernel
__global__ void array_sum(int *a, int *out, int n) {
    int i = threadIdx.x;
    int num_threads = blockDim.x;
    __shared__ int shared[15];

    for (int j = i; j < n; j += num_threads) shared[j] = a[j];

    for (int m = 8; m > 0; m >>= 1) {
        int step = 16 / m;
        int half_step = step / 2;
        __syncthreads();

        for (int j = i; j < m && j * step + half_step < n; j += num_threads) {
            shared[j * step] = shared[j * step] + shared[j * step + half_step];
        }
    }

    if (i == 0) out[0] = shared[0];
}


// Main Function
int main() {
    const int n = 15;
    int a[n], out[1];
    int *d_a, *d_out;

    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_out, sizeof(int));

    for(int i = 0; i < n; i++) a[i] = i;
    out[0] = 0;
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);

    array_sum<<<1, 8>>>(d_a, d_out, n);

    cudaMemcpy(out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_out);

    printf("sum of array: %d\n", out[0]);

    return 0;
}
