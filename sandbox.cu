#include<stdio.h>

__global__ void kernel(int *a, int *out) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    int local_x = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int shared[1024];
    shared[local_x] = a[local_x];
    out[local_x] += a[local_x];
    // if (x == 0) {
    //     for (int i = 0; i < 1024; i++) {
    //         printf("block %d, shared[%d]=%d\n", blockIdx.x, i, shared[i]);
    //     }
    // }
}

int main() {
    int n = 1024;
    int a[n], out[n];
    int *a_ptr, *out_ptr;
    cudaMalloc((void**)&a_ptr, n * sizeof(int));
    cudaMalloc((void**)&out_ptr, n * sizeof(int));

    for (int i = 0; i < n; i++) {
        a[i] = i;
        out[i] = 0;
    }

    cudaMemcpy(a_ptr, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_ptr, out, n * sizeof(int), cudaMemcpyHostToDevice);

    // dim3 threads(2, 2, 2);
    kernel<<<4, 4>>>(a_ptr, out_ptr);

    cudaMemcpy(out, out_ptr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(a_ptr);
    cudaFree(out_ptr);

    for (int i = 0; i < n; i++) {
        printf("%d ", out[i]);
    }
    printf("\n");

    return 0;
}
