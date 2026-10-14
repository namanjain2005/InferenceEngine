// Broadcast bias add for the cuBLAS linear path (cuBLAS has no fused bias).
// x is [batch, n] row-major, bias is [n]; done in place.
extern "C" __global__ void bias_add(
    float* __restrict__ x,
    const float* __restrict__ bias,
    int batch,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n;
    if (idx < total) x[idx] += bias[idx % n];
}
