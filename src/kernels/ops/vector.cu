extern "C" __global__ void vector_add(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" __global__ void embedding_lookup(const float* table, float* out, int token, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) out[i] = table[token * hidden + i];
}

extern "C" __global__ void apply_rope(
    const float* q, const float* k, float* q_out, float* k_out,
    int pos, int q_heads, int kv_heads, int dim, float theta) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int half = dim / 2;
    int q_n = q_heads * dim;
    int kv_n = kv_heads * dim;
    if (index < q_n) {
        int h = index / dim, d = index % dim;
        int pair = d < half ? d : d - half;
        float angle = (float)pos * powf(theta, -(2.0f * pair) / (float)dim);
        float c = cosf(angle), s = sinf(angle);
        int other = h * dim + (d < half ? d + half : d - half);
        q_out[index] = d < half ? q[index] * c - q[other] * s : q[index] * c + q[other] * s;
    }
    if (index < kv_n) {
        int h = index / dim, d = index % dim;
        int pair = d < half ? d : d - half;
        float angle = (float)pos * powf(theta, -(2.0f * pair) / (float)dim);
        float c = cosf(angle), s = sinf(angle);
        int other = h * dim + (d < half ? d + half : d - half);
        k_out[index] = d < half ? k[index] * c - k[other] * s : k[index] * c + k[other] * s;
    }
}

extern "C" __global__ void append_cache(
    const float* k, const float* v, float* k_cache, float* v_cache, int pos, int kv_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) {
        k_cache[pos * kv_dim + i] = k[i];
        v_cache[pos * kv_dim + i] = v[i];
    }
}
