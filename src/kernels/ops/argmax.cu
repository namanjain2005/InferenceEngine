extern "C" __global__ void argmax_f32(
    const float* __restrict__ values,
    int* __restrict__ result,
    int n) {
    __shared__ float best_values[256];
    __shared__ int best_indices[256];

    const int tid = threadIdx.x;
    float best_value = -3.402823466e+38F;
    int best_index = 2147483647;

    for (int index = tid; index < n; index += blockDim.x) {
        const float value = values[index];
        if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }

    best_values[tid] = best_value;
    best_indices[tid] = best_index;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            const float other_value = best_values[tid + offset];
            const int other_index = best_indices[tid + offset];
            if (other_value > best_values[tid]
                || (other_value == best_values[tid] && other_index < best_indices[tid])) {
                best_values[tid] = other_value;
                best_indices[tid] = other_index;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = best_indices[0] == 2147483647 ? 0 : best_indices[0];
    }
}

extern "C" __global__ void argmax_batch_f32(
    const float* __restrict__ values, int* __restrict__ result, int rows, int cols) {
    __shared__ float best_values[256];
    __shared__ int best_indices[256];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* input = values + row * cols;
    float best_value = -3.402823466e+38F;
    int best_index = 2147483647;
    for (int index = tid; index < cols; index += blockDim.x) {
        const float value = input[index];
        if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    best_values[tid] = best_value;
    best_indices[tid] = best_index;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            const float other_value = best_values[tid + offset];
            const int other_index = best_indices[tid + offset];
            if (other_value > best_values[tid]
                || (other_value == best_values[tid] && other_index < best_indices[tid])) {
                best_values[tid] = other_value;
                best_indices[tid] = other_index;
            }
        }
        __syncthreads();
    }
    if (tid == 0) result[row] = best_indices[0] == 2147483647 ? 0 : best_indices[0];
}
