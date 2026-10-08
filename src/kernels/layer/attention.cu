// Simple fused decode-step attention kernel for Grouped-Query Attention.
//
// One block per query head.  Within the block:
//   Phase 1 — threads cooperate to compute Q·K dot products (strided over seq positions).
//   Phase 2 — thread 0 runs softmax over the scores in shared memory.
//   Phase 3 — threads cooperate to compute the score-weighted V sum (strided over head_dim).
//
// Layout expected (matches the Rust KV-cache layout):
//   q       [num_q_heads * head_dim]           — query vector for current token
//   k_cache [seq_len     * kv_dim]             — all cached keys   (kv_dim = num_kv_heads * head_dim)
//   v_cache [seq_len     * kv_dim]             — all cached values
//   output  [num_q_heads * head_dim]           — attention output
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
    extern __shared__ float scores[];          // [seq_len]

    const int h    = blockIdx.x;               // query head index
    const int kv_h = h / heads_per_kv;         // corresponding kv head
    const int tid  = threadIdx.x;

    const float* q_head = q + h * head_dim;

    // ======== Phase 1: Q·K dot products ========
    for (int t = tid; t < seq_len; t += blockDim.x) {
        float dot = 0.0f;
        const float* k_ptr = k_cache + t * kv_dim + kv_h * head_dim;
        for (int d = 0; d < head_dim; d++)
            dot = fmaf(q_head[d], k_ptr[d], dot);
        scores[t] = dot * scale;
    }
    __syncthreads();

    // ======== Phase 2: Softmax (serial on thread 0 — simple, correct) ========
    if (tid == 0) {
        float max_val = -1e30f;
        for (int t = 0; t < seq_len; t++)
            max_val = fmaxf(max_val, scores[t]);

        float sum = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_val);
            sum += scores[t];
        }

        const float inv_sum = 1.0f / sum;
        for (int t = 0; t < seq_len; t++)
            scores[t] *= inv_sum;
    }
    __syncthreads();

    // ======== Phase 3: Score · V weighted sum ========
    float* out_head = output + h * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; t++)
            acc = fmaf(scores[t], v_cache[t * kv_dim + kv_h * head_dim + d], acc);
        out_head[d] = acc;
    }
}
