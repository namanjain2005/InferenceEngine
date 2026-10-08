#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,int K) {

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (row >= M || col >= K) {
        return;
    }

    float sum = 0.0f;
    for(int i=0;i<N;i++){
        sum += A[row*N + i] * B[col + i*K];
    }
    C[col + row*K] = sum;
}
