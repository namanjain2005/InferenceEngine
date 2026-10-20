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

__device__ __forceinline__ unsigned short float_to_half_bits(float value) {
    unsigned int bits = __float_as_uint(value);
    unsigned short sign = (unsigned short)((bits >> 16) & 0x8000u);
    int exp = (int)((bits >> 23) & 0xffu) - 127 + 15;
    unsigned int frac = bits & 0x7fffffu;
    if (((bits >> 23) & 0xffu) == 0xffu) return (unsigned short)(sign | (frac ? 0x7e00u : 0x7c00u));
    if (exp >= 31) return (unsigned short)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return sign;
        unsigned int mant = frac | 0x800000u;
        unsigned int shift = (unsigned int)(14 - exp);
        unsigned int rounded = (mant + (1u << (shift - 1)) - 1u + ((mant >> shift) & 1u)) >> shift;
        return (unsigned short)(sign | rounded);
    }
    unsigned int rounded = frac + 0xfffu + ((frac >> 13) & 1u);
    if (rounded & 0x800000u) { ++exp; rounded = 0; }
    if (exp >= 31) return (unsigned short)(sign | 0x7c00u);
    return (unsigned short)(sign | ((unsigned int)exp << 10) | ((rounded >> 13) & 0x3ffu));
}

extern "C" __global__ void cast_f32_to_f16(const float* input, unsigned short* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = float_to_half_bits(input[i]);
}

extern "C" __global__ void dequant_i8_rows(const signed char* input, const float* scales, float* output, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (i < total) output[i] = (float)input[i] * scales[i / cols];
}
