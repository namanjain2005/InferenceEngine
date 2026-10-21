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

extern "C" __global__ void embedding_batch(
    const float* table, const int* tokens, float* out, int batch, int hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden;
    if (idx < total) {
        int row = idx / hidden;
        int col = idx % hidden;
        out[idx] = table[tokens[row] * hidden + col];
    }
}

extern "C" __global__ void apply_rope_batch(
    const float* q, const float* k, float* q_out, float* k_out,
    int start_pos, int batch, int q_heads, int kv_heads, int dim, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = dim / 2;
    int q_n = q_heads * dim;
    int kv_n = kv_heads * dim;
    int total = batch * (q_n > kv_n ? q_n : kv_n);
    if (idx >= total) return;
    int row = -1, col = -1;
    // Map flat index to (row, col) over the wider of q/k widths.
    // Threads past the narrower width exit via the guards below.
    int width = q_n > kv_n ? q_n : kv_n;
    row = idx / width;
    col = idx % width;
    if (row >= batch) return;
    int pos = start_pos + row;
    if (col < q_n) {
        int h = col / dim, d = col % dim;
        int pair = d < half ? d : d - half;
        float angle = (float)pos * powf(theta, -(2.0f * pair) / (float)dim);
        float c = cosf(angle), s = sinf(angle);
        int other = h * dim + (d < half ? d + half : d - half);
        int base = row * q_n;
        q_out[base + col] = d < half ? q[base + col] * c - q[base + other] * s
                                     : q[base + col] * c + q[base + other] * s;
    }
    if (col < kv_n) {
        int h = col / dim, d = col % dim;
        int pair = d < half ? d : d - half;
        float angle = (float)pos * powf(theta, -(2.0f * pair) / (float)dim);
        float c = cosf(angle), s = sinf(angle);
        int other = h * dim + (d < half ? d + half : d - half);
        int base = row * kv_n;
        k_out[base + col] = d < half ? k[base + col] * c - k[base + other] * s
                                     : k[base + col] * c + k[base + other] * s;
    }
}

// Multi-request decode: each batch row can have a different absolute position.
extern "C" __global__ void apply_rope_batch_positions(
    const float* q, const float* k, const int* positions,
    float* q_out, float* k_out,
    int batch, int q_heads, int kv_heads, int dim, float theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = dim / 2;
    int q_n = q_heads * dim;
    int kv_n = kv_heads * dim;
    int width = q_n > kv_n ? q_n : kv_n;
    int total = batch * width;
    if (idx >= total) return;
    int row = idx / width;
    int col = idx % width;
    int pos = positions[row];
    if (col < q_n) {
        int h = col / dim, d = col % dim;
        int pair = d < half ? d : d - half;
        float angle = (float)pos * powf(theta, -(2.0f * pair) / (float)dim);
        float c = cosf(angle), s = sinf(angle);
        int other = h * dim + (d < half ? d + half : d - half);
        int base = row * q_n;
        q_out[base + col] = d < half ? q[base + col] * c - q[base + other] * s
                                     : q[base + col] * c + q[base + other] * s;
    }
    if (col < kv_n) {
        int h = col / dim, d = col % dim;
        int pair = d < half ? d : d - half;
        float angle = (float)pos * powf(theta, -(2.0f * pair) / (float)dim);
        float c = cosf(angle), s = sinf(angle);
        int other = h * dim + (d < half ? d + half : d - half);
        int base = row * kv_n;
        k_out[base + col] = d < half ? k[base + col] * c - k[base + other] * s
                                     : k[base + col] * c + k[base + other] * s;
    }
}

extern "C" __global__ void append_cache_batch(
    const float* k, const float* v, float* k_cache, float* v_cache,
    int start_pos, int batch, int kv_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * kv_dim;
    if (idx < total) {
        int row = idx / kv_dim;
        int col = idx % kv_dim;
        k_cache[(start_pos + row) * kv_dim + col] = k[idx];
        v_cache[(start_pos + row) * kv_dim + col] = v[idx];
    }
}

extern "C" __global__ void append_cache_batch_f16(
    const float* k, const float* v, unsigned short* k_cache, unsigned short* v_cache,
    int start_pos, int batch, int kv_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * kv_dim;
    if (idx < total) {
        int row = idx / kv_dim;
        int col = idx % kv_dim;
        int cache_idx = (start_pos + row) * kv_dim + col;
        unsigned short key_half;
        unsigned short value_half;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(key_half) : "f"(k[idx]));
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(value_half) : "f"(v[idx]));
        k_cache[cache_idx] = key_half;
        v_cache[cache_idx] = value_half;
    }
}

extern "C" __global__ void gather_row(
    const float* input, float* out, int row, int batch, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width && row < batch) out[i] = input[row * width + i];
}
