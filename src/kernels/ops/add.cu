#include <cuda_runtime.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int N) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < N*N){
        C[idx] = A[idx] + B[idx];
    }
}
