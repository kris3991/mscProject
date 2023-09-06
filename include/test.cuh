#include <cuda_runtime.h>

__global__ void vectorAdd(int *a, int *b, int *c, int n);

void test(int* c_i, int n);