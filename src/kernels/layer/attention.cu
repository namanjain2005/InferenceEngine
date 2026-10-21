// Tiled online-softmax attention for causal decode and prefill. Each CTA owns
// one query head, streams bounded K/V tiles, and keeps the running softmax
// statistics, tile scores, and output accumulator on chip. The sequence-sized
// score and probability buffers used by the previous implementation are not
// needed.
constexpr int WARP_SIZE = 32;

template <typename T>
__device__ float attention_to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ float attention_to_float<unsigned short>(unsigned short value) {
    float converted;
    asm("cvt.f32.f16 %0, %1;" : "=f"(converted) : "h"(value));
    return converted;
}

template <typename T>
__device__ T attention_from_float(float value) {
    return static_cast<T>(value);
}

template <>
__device__ unsigned short attention_from_float<unsigned short>(float value) {
    unsigned short converted;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(converted) : "f"(value));
    return converted;
}

template <int TILE_SIZE, typename T>
__device__ void attention_flash_one_query(
    const T* __restrict__ q,
    const T* __restrict__ k_cache,
    const T* __restrict__ v_cache,
    T*       __restrict__ output,
    int query_row,
    int query_head,
    int num_query_heads,
    int seq_len,
    int kv_dim,
    int head_dim,
    int heads_per_kv,
    float scale,
    float* shared)
{
    float* scores = shared;
    float* reduction = scores + TILE_SIZE;
    float* output_acc = reduction + 1;

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int kv_head = query_head / heads_per_kv;
    const int row_offset = query_row * num_query_heads * head_dim;
    const T* q_head = q + row_offset + query_head * head_dim;
    float running_max = -1.0e30f;
    float running_sum = 0.0f;

    for (int d = tid; d < head_dim; d += blockDim.x)
        output_acc[d] = 0.0f;
    __syncthreads();

    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
        const int remaining = seq_len - tile_start;
        const int tile_len = remaining < TILE_SIZE ? remaining : TILE_SIZE;

        // A warp computes each Q.K score, reducing over contiguous dimensions.
        for (int j = warp; j < tile_len; j += warps_per_block) {
            float dot = 0.0f;
            const T* k_row = k_cache + (tile_start + j) * kv_dim + kv_head * head_dim;
            for (int d = lane; d < head_dim; d += WARP_SIZE)
                dot = fmaf(attention_to_float(q_head[d]), attention_to_float(k_row[d]), dot);
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            if (lane == 0)
                scores[j] = dot * scale;
        }
        __syncthreads();

        // One warp reduces strided partials for the tile's scores. This avoids
        // a 256-thread block reduction and its barriers for every key tile.
        if (tid < WARP_SIZE) {
            float local_max = -1.0e30f;
            for (int j = tid; j < tile_len; j += WARP_SIZE)
                local_max = fmaxf(local_max, scores[j]);
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                local_max = fmaxf(
                    local_max,
                    __shfl_down_sync(0xffffffff, local_max, offset));
            if (tid == 0)
                reduction[0] = local_max;
        }
        __syncthreads();
        const float tile_max = reduction[0];
        // All threads read tile_max before reduction[] is reused for the sum.
        __syncthreads();

        for (int j = tid; j < tile_len; j += blockDim.x) {
            scores[j] = expf(scores[j] - tile_max);
        }
        __syncthreads();
        if (tid < WARP_SIZE) {
            float local_sum = 0.0f;
            for (int j = tid; j < tile_len; j += WARP_SIZE)
                local_sum += scores[j];
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
            if (tid == 0)
                reduction[0] = local_sum;
        }
        __syncthreads();

        const float tile_sum = reduction[0];
        const float next_max = fmaxf(running_max, tile_max);
        const float old_scale = expf(running_max - next_max);
        const float tile_scale = expf(tile_max - next_max);
        const float next_sum = old_scale * running_sum + tile_scale * tile_sum;

        for (int d = tid; d < head_dim; d += blockDim.x) {
            float acc = old_scale * output_acc[d];
            for (int j = 0; j < tile_len; ++j)
                acc = fmaf(
                    tile_scale * scores[j],
                    attention_to_float(v_cache[(tile_start + j) * kv_dim + kv_head * head_dim + d]),
                    acc);
            output_acc[d] = acc;
        }
        running_max = next_max;
        running_sum = next_sum;
        // Finish reading scores before reusing their shared-memory storage.
        __syncthreads();
    }

    const int output_offset = query_row * num_query_heads * head_dim + query_head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x)
        output[output_offset + d] = attention_from_float<T>(output_acc[d] / running_sum);
}

#define DEFINE_FLASH_DECODE(TILE) \
extern "C" __global__ void attention_decode_flash_##TILE( \
    const float* q, const float* k_cache, const float* v_cache, float* output, \
    int seq_len, int kv_dim, int head_dim, int heads_per_kv, float scale) \
{ \
    extern __shared__ float shared[]; \
    const int h = blockIdx.x; \
    attention_flash_one_query<TILE, float>(q, k_cache, v_cache, output, 0, h, (int)gridDim.x, \
        seq_len, kv_dim, head_dim, heads_per_kv, scale, shared); \
}

#define DEFINE_FLASH_PREFILL(TILE) \
extern "C" __global__ void attention_prefill_flash_##TILE( \
    const float* q, const float* k_cache, const float* v_cache, float* output, \
    int start_pos, int batch, int num_q_heads, int kv_dim, int head_dim, \
    int heads_per_kv, float scale) \
{ \
    extern __shared__ float shared[]; \
    const int row = blockIdx.x; \
    const int h = blockIdx.y; \
    if (row >= batch) return; \
    const int seq_len = start_pos + row + 1; \
    attention_flash_one_query<TILE, float>(q, k_cache, v_cache, output, row, h, num_q_heads, \
        seq_len, kv_dim, head_dim, heads_per_kv, scale, shared); \
}

DEFINE_FLASH_DECODE(256)
DEFINE_FLASH_DECODE(128)
DEFINE_FLASH_DECODE(64)
DEFINE_FLASH_DECODE(32)
DEFINE_FLASH_DECODE(16)
DEFINE_FLASH_DECODE(8)
DEFINE_FLASH_DECODE(4)
DEFINE_FLASH_PREFILL(256)
DEFINE_FLASH_PREFILL(128)
DEFINE_FLASH_PREFILL(64)
DEFINE_FLASH_PREFILL(32)
DEFINE_FLASH_PREFILL(16)
DEFINE_FLASH_PREFILL(8)
DEFINE_FLASH_PREFILL(4)

#define DEFINE_FLASH_DECODE_F16(TILE) \
extern "C" __global__ void attention_decode_flash_f16_##TILE( \
    const unsigned short* q, const unsigned short* k_cache, const unsigned short* v_cache, unsigned short* output, \
    int seq_len, int kv_dim, int head_dim, int heads_per_kv, float scale) \
{ \
    extern __shared__ float shared[]; \
    const int h = blockIdx.x; \
    attention_flash_one_query<TILE, unsigned short>(q, k_cache, v_cache, output, 0, h, (int)gridDim.x, \
        seq_len, kv_dim, head_dim, heads_per_kv, scale, shared); \
}

#define DEFINE_FLASH_PREFILL_F16(TILE) \
extern "C" __global__ void attention_prefill_flash_f16_##TILE( \
    const unsigned short* q, const unsigned short* k_cache, const unsigned short* v_cache, unsigned short* output, \
    int start_pos, int batch, int num_q_heads, int kv_dim, int head_dim, \
    int heads_per_kv, float scale) \
{ \
    extern __shared__ float shared[]; \
    const int row = blockIdx.x; \
    const int h = blockIdx.y; \
    if (row >= batch) return; \
    const int seq_len = start_pos + row + 1; \
    attention_flash_one_query<TILE, unsigned short>(q, k_cache, v_cache, output, row, h, num_q_heads, \
        seq_len, kv_dim, head_dim, heads_per_kv, scale, shared); \
}

DEFINE_FLASH_DECODE_F16(256)
DEFINE_FLASH_DECODE_F16(128)
DEFINE_FLASH_DECODE_F16(64)
DEFINE_FLASH_DECODE_F16(32)
DEFINE_FLASH_DECODE_F16(16)
DEFINE_FLASH_DECODE_F16(8)
DEFINE_FLASH_DECODE_F16(4)
DEFINE_FLASH_PREFILL_F16(256)
DEFINE_FLASH_PREFILL_F16(128)
DEFINE_FLASH_PREFILL_F16(64)
DEFINE_FLASH_PREFILL_F16(32)
DEFINE_FLASH_PREFILL_F16(16)
DEFINE_FLASH_PREFILL_F16(8)
DEFINE_FLASH_PREFILL_F16(4)

extern "C" __global__ void cast_f32_to_f16(const float* input, unsigned short* output, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned short converted;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(converted) : "f"(input[idx]));
        output[idx] = converted;
    }
}

extern "C" __global__ void cast_f16_to_f32(const unsigned short* input, float* output, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float converted;
        asm("cvt.f32.f16 %0, %1;" : "=f"(converted) : "h"(input[idx]));
        output[idx] = converted;
    }
}
