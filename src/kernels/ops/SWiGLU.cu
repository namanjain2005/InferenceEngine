extern "C" __global__ void SWiGLU(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    int halfN) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < halfN) {
       float x = gate[index];
       output[index] = x * (1.0f / (1.0f + expf(-x))) * up[index];
    }
}
