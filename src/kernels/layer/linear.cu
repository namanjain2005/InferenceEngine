constexpr int WARP_SIZE = 32;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
constexpr int INPUT_TILE_SIZE = THREADS_PER_BLOCK;

extern "C" __global__ void linear(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_features,
    int out_features) {
    __shared__ float input_tile[INPUT_TILE_SIZE];

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int output_blocks_per_row = (out_features - 1) / WARPS_PER_BLOCK + 1;
    const int row = blockIdx.x / output_blocks_per_row;
    const int output_base = (blockIdx.x % output_blocks_per_row) * WARPS_PER_BLOCK;
    const int out = output_base + warp;

    const float* input_row = input + row * in_features;
    float partial_sum = 0.0f;

    // Stage each input tile once, then reuse it across the output rows handled
    // by this block. Weight loads stay coalesced within each warp.
    for (int k_base = 0; k_base < in_features; k_base += INPUT_TILE_SIZE) {
        const int input_index = k_base + tid;
        input_tile[tid] = input_index < in_features ? input_row[input_index] : 0.0f;
        __syncthreads();

        if (out < out_features) {
            const int remaining = in_features - k_base;
            const int tile_size = remaining < INPUT_TILE_SIZE ? remaining : INPUT_TILE_SIZE;
            const float* weight_row = weights + out * in_features + k_base;
            for (int k = lane; k < tile_size; k += WARP_SIZE)
                partial_sum = fmaf(weight_row[k], input_tile[k], partial_sum);
        }
        __syncthreads();
    }

    if (out < out_features) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);

        if (lane == 0)
            output[row * out_features + out] = partial_sum + bias[out];
    }
}
