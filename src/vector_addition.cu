#include "../include/test.cuh"

// CUDA kernel to add two vectors
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void test(int* c_i, int N)
{
   const int n = 1000; // Size of the vectors
    int *a, *b, *c; // Host vectors
    int *d_a, *d_b, *d_c; // Device vectors

    // Allocate memory on the host
    a = new int[n];
    b = new int[n];

    // Initialize vectors a and b
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_b, n * sizeof(int));
    cudaMalloc((void**)&d_c, n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with 1 block and 256 threads per block
    vectorAdd<<<1, n>>>(d_a, d_b, d_c, n);

    // Copy the result back from device to host
    cudaMemcpy(c_i, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    

    // Free host memory
    delete[] a;
    delete[] b;
}

