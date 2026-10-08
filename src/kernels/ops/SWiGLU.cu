extern "C" __global__ void SWiGLU(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int halfN) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * halfN;
    if(index < total){
       float x1 = input[index];
       float x2 = input[index + halfN];
       output[index] = x1 * (1.0f/(1.0f + __exp(-x1))) * x2;
    } 
}
