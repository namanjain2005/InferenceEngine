// RMSNorm kernel for single-token inference (one row at a time).
// Launched with 1 block of THREADS threads.
// Uses shared memory parallel reduction to compute sum-of-squares,
// then each thread normalizes + scales its assigned elements.
//
// input  [n]   — hidden state
// weight [n]   — learned per-element scale
// output [n]   — normalized result
extern "C" __global__ void rmsnorm(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int   n,
    float eps)
{
    extern __shared__ float shared[];          // blockDim.x floats

    const int tid = threadIdx.x;

    // ---- Step 1: partial sum of squares ----
    float local_ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = input[i];
        local_ss = fmaf(v, v, local_ss);       // v*v + acc
    }
    shared[tid] = local_ss;
    __syncthreads();

    // ---- Step 2: tree reduction ----
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s)
            shared[tid] += shared[tid + s];
        __syncthreads();
    }

    // ---- Step 3: compute inverse RMS (broadcast via shared[0]) ----
    if (tid == 0)
        shared[0] = rsqrtf(shared[0] / (float)n + eps);
    __syncthreads();

    const float inv_rms = shared[0];

    // ---- Step 4: normalize & scale ----
    for (int i = tid; i < n; i += blockDim.x)
        output[i] = input[i] * inv_rms * weight[i];
}
