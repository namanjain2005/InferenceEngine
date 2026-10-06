extern "C" __global__ void linear(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_features,
    int out_features) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * out_features;
    if (index >= total) return;

    const int row = index / out_features;
    const int out = index % out_features;
    const float* x = input + row * in_features;
    const float* w = weights + out * in_features;
    float sum = bias[out];
    for (int i = 0; i < in_features; ++i) {
        sum = fmaf(w[i], x[i], sum);
    }
    output[index] = sum;
}
