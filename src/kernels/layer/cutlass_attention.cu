#include "kernel_forward.h"

// CUTLASS 3.9.2 fused multi-head attention, configured for Volta tensor cores.
// The batch hook maps each query head to its grouped-query KV head.
struct GroupedQueryBatchHook {
  template <typename Params>
  CUTLASS_DEVICE static bool advance_to_batch(
      Params& p, int64_t& /*q_start*/, int64_t& /*k_start*/) {
    const int query_head = blockIdx.y;
    const int heads_per_kv = static_cast<int>(p.k_strideB);
    const int kv_head = query_head / heads_per_kv;
    p.key_ptr += static_cast<int64_t>(kv_head) * p.head_dim;
    p.value_ptr += static_cast<int64_t>(kv_head) * p.head_dim_value;
    return true;
  }
};

using CutlassSm70Fmha = AttentionKernel<
    cutlass::half_t,
    cutlass::arch::Sm70,
    true,
    32,
    64,
    128,
    false,
    false,
    GroupedQueryBatchHook>;

extern "C" __device__ __constant__ unsigned int cutlass_attention_shared_bytes =
    sizeof(CutlassSm70Fmha::SharedStorage);

extern "C" __global__ __launch_bounds__(
    CutlassSm70Fmha::kNumThreads,
    CutlassSm70Fmha::kMinBlocksPerSm)
void cutlass_fmha_prefill_f16(
    const cutlass::half_t* q,
    const cutlass::half_t* k,
    const cutlass::half_t* v,
    cutlass::half_t* output,
    float* output_accum,
    int start_pos,
    int batch,
    int num_q_heads,
    int num_kv_heads,
    int head_dim) {
  CutlassSm70Fmha::Params p{};
  p.query_ptr = const_cast<cutlass::half_t*>(q);
  p.key_ptr = const_cast<cutlass::half_t*>(k);
  p.value_ptr = const_cast<cutlass::half_t*>(v);
  p.output_ptr = output;
  p.output_accum_ptr = output_accum;
  p.scale = rsqrtf(static_cast<float>(head_dim));
  p.head_dim = head_dim;
  p.head_dim_value = head_dim;
  p.num_queries = batch;
  p.num_keys = start_pos + batch;
  p.custom_mask_type = CutlassSm70Fmha::CausalFromBottomRight;
  p.causal_diagonal_offset = start_pos;
  p.q_strideM = num_q_heads * head_dim;
  p.k_strideM = num_kv_heads * head_dim;
  p.v_strideM = num_kv_heads * head_dim;
  p.o_strideM = num_q_heads * head_dim;
  p.q_strideH = head_dim;
  p.k_strideH = 0;
  p.v_strideH = 0;
  // The custom hook uses this otherwise-unused stride slot to map the query
  // head index to its grouped-query KV head.
  p.k_strideB = num_q_heads / num_kv_heads;
  p.num_batches = 1;
  p.num_heads = num_q_heads;

  if (!p.advance_to_block()) {
    return;
  }
  CutlassSm70Fmha::attention_kernel(p);
}
