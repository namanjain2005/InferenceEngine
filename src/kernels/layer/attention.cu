// CUDA-core online-softmax attention for decode and the FP32 fallback. Each
// CTA owns one query head and streams bounded K/V tiles.
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

// V100-compatible, Tensor Core FlashAttention-style prefill. A CTA owns four
// query positions for every query head sharing one KV head. It stages one
// 32-key K/V tile, reuses it across those query rows/heads, computes QK and PV
// with FP16 WMMA, and updates the softmax and output online. The output stays
// in shared memory until the complete key sequence has been consumed.
#include <mma.h>

extern "C" __global__ void attention_prefill_flash_gqa_f16(
    const unsigned short* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    unsigned short* __restrict__ output,
    int start_pos,
    int batch,
    int num_q_heads,
    int num_kv_heads,
    int head_dim) {
  using namespace nvcuda;
  constexpr int kQueryRows = 32;
  constexpr int kKeyTile = 32;
  constexpr int kMaxHeadDim = 128;
  constexpr int kQueryTileRows = 4;

  __shared__ __align__(16) __half q_tile[kQueryRows * kMaxHeadDim];
  // This buffer holds K in row-major form for QK, then V transposed for PV.
  // Reusing it keeps the complete CTA footprint below V100's shared-memory
  // limit without rereading either tensor from global memory.
  __shared__ __align__(16) __half kv_tile[kKeyTile * kMaxHeadDim];
  __shared__ __align__(16) float scores[kQueryRows * kKeyTile];
  __shared__ __align__(16) __half probabilities[kQueryRows * kKeyTile];
  __shared__ __align__(16) float output_acc[kQueryRows * kMaxHeadDim];
  __shared__ float row_max[kQueryRows];
  __shared__ float row_sum[kQueryRows];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int query_start = blockIdx.x * kQueryTileRows;
  const int kv_head = blockIdx.y;
  const int heads_per_kv = num_q_heads / num_kv_heads;
  const int active_rows = min(kQueryRows, kQueryTileRows * heads_per_kv);
  const int kv_dim = num_kv_heads * head_dim;
  const int total_kv_len = start_pos + batch;
  const int query_kv_limit = start_pos + min(batch, query_start + kQueryTileRows);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  // Pack four query positions and all heads in this GQA group into WMMA rows.
  for (int index = tid; index < kQueryRows * kMaxHeadDim; index += blockDim.x) {
    const int row = index / kMaxHeadDim;
    const int dim = index % kMaxHeadDim;
    const int query_row = query_start + row / heads_per_kv;
    const int query_head = kv_head * heads_per_kv + row % heads_per_kv;
    if (row < active_rows && query_row < batch && dim < head_dim) {
      const int offset = (query_row * num_q_heads + query_head) * head_dim + dim;
      q_tile[index] = reinterpret_cast<const __half*>(q)[offset];
    } else {
      q_tile[index] = __float2half(0.0f);
    }
  }
  for (int index = tid; index < kQueryRows * kMaxHeadDim; index += blockDim.x)
    output_acc[index] = 0.0f;
  for (int row = tid; row < kQueryRows; row += blockDim.x) {
    row_max[row] = -1.0e30f;
    row_sum[row] = 0.0f;
  }
  __syncthreads();

  for (int tile_start = 0; tile_start < query_kv_limit; tile_start += kKeyTile) {
    for (int index = tid; index < kKeyTile * kMaxHeadDim; index += blockDim.x) {
      const int key = index / kMaxHeadDim;
      const int dim = index % kMaxHeadDim;
      const int key_pos = tile_start + key;
      if (key_pos < total_kv_len && dim < head_dim) {
        const int offset = key_pos * kv_dim + kv_head * head_dim + dim;
        kv_tile[index] = reinterpret_cast<const __half*>(k_cache)[offset];
      } else {
        kv_tile[index] = __float2half(0.0f);
      }
    }
    __syncthreads();

    // Four warps calculate two 16-row halves by two 16-key halves.
    if (warp < 4) {
      const int row_block = warp / 2;
      const int key_offset = (warp % 2) * 16;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.0f);
      for (int dim = 0; dim < head_dim; dim += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
        wmma::load_matrix_sync(a, q_tile + row_block * 16 * kMaxHeadDim + dim, kMaxHeadDim);
        wmma::load_matrix_sync(b, kv_tile + key_offset * kMaxHeadDim + dim, kMaxHeadDim);
        wmma::mma_sync(acc, a, b, acc);
      }
      wmma::store_matrix_sync(
          scores + row_block * 16 * kKeyTile + key_offset,
          acc,
          kKeyTile,
          wmma::mem_row_major);
    }
    __syncthreads();

    // A warp reduces one query row at a time and handles four rows per tile.
    for (int row = warp; row < kQueryRows; row += 8) {
      const int query_row = query_start + row / heads_per_kv;
      const bool active = row < active_rows && query_row < batch;
      const int query_pos = start_pos + query_row;
      float score = scores[row * kKeyTile + lane] * scale;
      const int key_pos = tile_start + lane;
      const bool valid = active && key_pos < total_kv_len && key_pos <= query_pos;
      if (!valid) score = -1.0e30f;
      scores[row * kKeyTile + lane] = score;
      float tile_max = score;
      for (int offset = 16; offset > 0; offset >>= 1)
        tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, offset));
      tile_max = __shfl_sync(0xffffffff, tile_max, 0);
      const float old_max = row_max[row];
      const float next_max = fmaxf(old_max, tile_max);
      const float alpha = expf(old_max - next_max);
      const float probability = valid ? expf(score - next_max) : 0.0f;
      float tile_sum = probability;
      for (int offset = 16; offset > 0; offset >>= 1)
        tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);
      tile_sum = __shfl_sync(0xffffffff, tile_sum, 0);
      __syncwarp();
      if (lane == 0) {
        row_max[row] = next_max;
        row_sum[row] = alpha * row_sum[row] + tile_sum;
      }
      probabilities[row * kKeyTile + lane] = __float2half(probability);
      for (int dim = lane; dim < kMaxHeadDim; dim += WARP_SIZE)
        output_acc[row * kMaxHeadDim + dim] *= alpha;
    }
    __syncthreads();

    // The QK phase is complete, so reuse K's shared-memory allocation for
    // transposed V. Each element is still loaded from global memory once.
    for (int index = tid; index < kKeyTile * kMaxHeadDim; index += blockDim.x) {
      const int key = index / kMaxHeadDim;
      const int dim = index % kMaxHeadDim;
      const int key_pos = tile_start + key;
      if (key_pos < total_kv_len && dim < head_dim) {
        const int offset = key_pos * kv_dim + kv_head * head_dim + dim;
        kv_tile[dim * kKeyTile + key] = reinterpret_cast<const __half*>(v_cache)[offset];
      } else {
        kv_tile[dim * kKeyTile + key] = __float2half(0.0f);
      }
    }
    __syncthreads();

    // Eight warps own output-column tiles; each handles both query-row halves.
    if (warp < kMaxHeadDim / 16) {
      const int dim_offset = warp * 16;
      for (int row_block = 0; row_block < 2; ++row_block) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::load_matrix_sync(
            acc,
            output_acc + row_block * 16 * kMaxHeadDim + dim_offset,
            kMaxHeadDim,
            wmma::mem_row_major);
        for (int key_offset = 0; key_offset < kKeyTile; key_offset += 16) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
          wmma::load_matrix_sync(
              a,
              probabilities + row_block * 16 * kKeyTile + key_offset,
              kKeyTile);
          wmma::load_matrix_sync(b, kv_tile + dim_offset * kKeyTile + key_offset, kKeyTile);
          wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(
            output_acc + row_block * 16 * kMaxHeadDim + dim_offset,
            acc,
            kMaxHeadDim,
            wmma::mem_row_major);
      }
    }
    __syncthreads();
  }

  for (int index = tid; index < kQueryRows * kMaxHeadDim; index += blockDim.x) {
    const int row = index / kMaxHeadDim;
    const int dim = index % kMaxHeadDim;
    const int query_row = query_start + row / heads_per_kv;
    if (row < active_rows && query_row < batch && dim < head_dim) {
      const int query_head = kv_head * heads_per_kv + row % heads_per_kv;
      const int offset = (query_row * num_q_heads + query_head) * head_dim + dim;
      output_acc[index] /= row_sum[row];
      reinterpret_cast<__half*>(output)[offset] = __float2half_rn(output_acc[index]);
    }
  }
}

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
