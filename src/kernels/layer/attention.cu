// Simple fused decode-step attention kernel for Grouped-Query Attention.
//
// One block per query head.  Within the block:
//   Phase 1 - threads cooperate to compute Q*K dot products (strided over seq positions).
//   Phase 2 - threads parallelize the softmax over the scores in shared memory.
//   Phase 3 - threads cooperate to compute the score-weighted V sum (strided over head_dim).
//
// Layout expected (matches the Rust KV-cache layout):
//   q       [num_q_heads * head_dim] - query vector for current token
//   k_cache [seq_len     * kv_dim] - all cached keys (kv_dim = num_kv_heads * head_dim)
//   v_cache [seq_len     * kv_dim] - all cached values
//   output  [num_q_heads * head_dim] - attention output
//
// Dynamic shared memory: seq_len * sizeof(float) for the per-head score buffer.
extern "C" __global__ void attention_decode(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float*       __restrict__ output,
    int   seq_len,
    int   kv_dim,
    int   head_dim,
    int   heads_per_kv,
    float scale)
{
    extern __shared__ float scores[];
    __shared__ float reduction[256];

    const int h    = blockIdx.x;
    const int kv_h = h / heads_per_kv;
    const int tid  = threadIdx.x;

    const float* q_head = q + h * head_dim;

    // ======== Phase 1: Q*K dot products ========
    for (int t = tid; t < seq_len; t += blockDim.x) {
        float dot = 0.0f;
        const float* k_ptr = k_cache + t * kv_dim + kv_h * head_dim;
        for (int d = 0; d < head_dim; d++)
            dot = fmaf(q_head[d], k_ptr[d], dot);
        scores[t] = dot * scale;
    }
    __syncthreads();

    // ======== Phase 2: Parallel stable softmax ========
    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += blockDim.x)
        local_max = fmaxf(local_max, scores[t]);
    reduction[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
        __syncthreads();
    }
    const float max_val = reduction[0];

    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        scores[t] = expf(scores[t] - max_val);
        local_sum += scores[t];
    }
    reduction[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    const float inv_sum = 1.0f / reduction[0];

    for (int t = tid; t < seq_len; t += blockDim.x)
        scores[t] *= inv_sum;
    __syncthreads();

    // ======== Phase 3: Score*V weighted sum ========
    float* out_head = output + h * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; t++)
            acc = fmaf(scores[t], v_cache[t * kv_dim + kv_h * head_dim + d], acc);
        out_head[d] = acc;
    }
}
