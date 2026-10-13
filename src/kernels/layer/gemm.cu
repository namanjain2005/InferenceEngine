// Blocked SGEMM for prefill: C[M,N] = A[M,K] * W^T + bias, where W is [N,K]
// row-major. Same weight layout as the GEMV baseline in linear.cu
// (weights[n * K + k]), so both paths share the exact same device slices.
//
// Tiling (tuned for V100, f32):
//   BM=32 rows x BN=128 cols per block, BK=32 reduction steps.
//   256 threads/block = 8 row-groups x 32 col-groups.
//   Each thread owns a TM=4 x TN=4 register tile (16 accumulators).
//   Shared: As[32*32] + Bs[32*128] = 5120 floats (~20KB).
//
// The inner loop hoists each As entry into a register and reuses it across
// the TN outputs (the Btmp-style reuse from the reference snippet, mirrored
// here on the A side since our B/W tile is transposed relative to vanilla
// GEMM: each A element is reused TN=4 times, each W element TM=4 times).

constexpr int BM = 32;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS = 256;
constexpr int ROW_GROUPS = BM / TM;  // 8
constexpr int COL_GROUPS = BN / TN;  // 32

extern "C" __global__ void gemm(
    const float* __restrict__ A,       // [M, K]
    const float* __restrict__ W,       // [N, K]
    const float* __restrict__ bias,    // [N]
    float* __restrict__ C,             // [M, N]
    int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int tid = threadIdx.x;
    const int threadRow = tid / COL_GROUPS;  // 0..8
    const int threadCol = tid % COL_GROUPS;  // 0..32

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    // Register tile: TM rows x TN cols.
    float acc[TM * TN];
#pragma unroll
    for (int i = 0; i < TM * TN; ++i) acc[i] = 0.0f;

    const int numKTiles = (K + BK - 1) / BK;
    for (int t = 0; t < numKTiles; ++t) {
        const int kTile = t * BK;

        // Cooperative strided load of both tiles (5120 elems / 256 threads).
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            int r = idx / BK;
            int c = idx % BK;
            int gRow = blockRow + r;
            int gK = kTile + c;
            As[idx] = (gRow < M && gK < K) ? A[gRow * K + gK] : 0.0f;
        }
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            int r = idx / BN;
            int c = idx % BN;
            int gCol = blockCol + c;
            int gK = kTile + r;
            Bs[idx] = (gCol < N && gK < K) ? W[gCol * K + gK] : 0.0f;
        }
        __syncthreads();

        // Compute: outer loop over BK, inner over register tile.
        // Each As value is reused across TN columns; each Bs value across TM rows.
        for (int k = 0; k < BK; ++k) {
            float aReg[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                aReg[i] = As[(threadRow * TM + i) * BK + k];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                float bVal = Bs[k * BN + threadCol * TN + j];
#pragma unroll
                for (int i = 0; i < TM; ++i)
                    acc[i * TN + j] = fmaf(aReg[i], bVal, acc[i * TN + j]);
            }
        }
        __syncthreads();
    }

    // Write back with bias.
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gRow = blockRow + threadRow * TM + i;
        if (gRow >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gCol = blockCol + threadCol * TN + j;
            if (gCol < N) C[gRow * N + gCol] = acc[i * TN + j] + bias[gCol];
        }
    }
}
