use super::LinearOp;
use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

const BIAS_CUDA: &str = include_str!("layer/bias.cu");

/// cuBLAS linear backend sharing the exact same row-major weight layout as
/// the custom kernels: input `[batch, K]`, weights `[N, K]`,
/// output `[batch, N]` with `out[m, n] = sum_k A[m, k] * W[n, k] + bias[n]`.
///
/// cuBLAS assumes column-major, so both entry points use the standard
/// row-major mapping: each buffer viewed as the transpose of its logical
/// shape (weights-as-[K, N], input-as-[K, batch]), with transa/transb set.
/// No data rearrangement; verified element-wise against the custom kernels.
pub struct CublasLinear {
    blas: CudaBlas,
    _bias_module: Arc<CudaModule>,
    bias_fn: CudaFunction,
}

impl CublasLinear {
    pub fn new(context: &Arc<CudaContext>, stream: Arc<CudaStream>) -> Result<Self, String> {
        let blas = CudaBlas::new(stream).map_err(|e| format!("Creating cuBLAS handle failed: {e:?}"))?;
        let ptx = compile_ptx(BIAS_CUDA).map_err(|e| format!("Bias CUDA compilation failed: {e}"))?;
        let module = context.load_module(ptx).map_err(|e| format!("Loading bias CUDA module failed: {e}"))?;
        let bias_fn = module.load_function("bias_add").map_err(|e| format!("Loading bias_add function failed: {e}"))?;
        Ok(Self { blas, _bias_module: module, bias_fn })
    }

    fn check(
        input_len: usize,
        weight_len: usize,
        bias_len: Option<usize>,
        batch: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<(), String> {
        if batch == 0 || in_features == 0 || out_features == 0 {
            return Err("cuBLAS linear dimensions must be non-zero".into());
        }
        let input_len_exp = batch.checked_mul(in_features).ok_or("cuBLAS linear input size overflow")?;
        let weight_len_exp = out_features.checked_mul(in_features).ok_or("cuBLAS linear weight size overflow")?;
        if input_len != input_len_exp || weight_len != weight_len_exp
            || bias_len.map_or(false, |b| b != out_features)
        {
            return Err(format!("cuBLAS linear device shape mismatch: input {input_len}, weights {weight_len}, bias {bias_len:?}; expected {input_len_exp}, {weight_len_exp}, {out_features}"));
        }
        if [batch, in_features, out_features].iter().any(|&v| v > i32::MAX as usize) {
            return Err("cuBLAS linear dimensions exceed index range".into());
        }
        Ok(())
    }

    fn add_bias(
        &self,
        stream: &Arc<CudaStream>,
        output: &mut CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
        out_features: usize,
    ) -> Result<(), String> {
        let Some(bias) = bias else { return Ok(()) };
        let (batch_i32, n_i32) = (batch as i32, out_features as i32);
        let mut launch = stream.launch_builder(&self.bias_fn);
        launch.arg(output);
        launch.arg(bias);
        launch.arg(&batch_i32);
        launch.arg(&n_i32);
        let total = batch.checked_mul(out_features).ok_or("cuBLAS bias size overflow")?;
        let grid = total.div_ceil(256) as u32;
        unsafe { launch.launch(LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }) }
            .map_err(|e| format!("Launching bias_add kernel failed: {e}"))?;
        Ok(())
    }
}

impl LinearOp for CublasLinear {
    fn upload(&self, stream: &Arc<CudaStream>, values: &[f32], name: &str) -> Result<CudaSlice<f32>, String> {
        let device = stream.clone_htod(values).map_err(|e| format!("Uploading {name} to CUDA failed: {e}"))?;
        stream.synchronize().map_err(|e| format!("Waiting for {name} upload failed: {e}"))?;
        Ok(device)
    }

    fn gemv(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, String> {
        // NOTE: deliberately routed through sgemm with batch=1 instead of
        // cublasSgemv: cublas 12.4's sgemv-transpose path writes past the
        // y buffer for K != N shapes (compute-sanitizer: Invalid __global__
        // write in gemv2T_kernel; dmesg Xid 31 PDE faults), heap-corrupting
        // later calls. sgemm with M=1 is sanitizer-clean and value-exact.
        self.gemm(stream, input, weights, bias, 1, in_features, out_features)
    }

    fn gemm(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, String> {
        Self::check(input.len(), weights.len(), bias.map(|b| b.len()), batch, in_features, out_features)?;
        let output_len = batch * out_features;
        let mut output = stream.alloc_zeros::<f32>(output_len).map_err(|e| format!("Allocating cuBLAS gemm output failed: {e}"))?;
        // Row-major C[M, N] = A[M, K] * Wt[K, N] as column-major
        // Cc[N, M] = op(W) * op(A): W-as-[K, N] transposed to NxK,
        // input-as-[K, M] used as-is (= A^T, already KxM).
        let cfg = GemmConfig {
            transa: sys::cublasOperation_t::CUBLAS_OP_T,
            transb: sys::cublasOperation_t::CUBLAS_OP_N,
            m: out_features as i32,
            n: batch as i32,
            k: in_features as i32,
            alpha: 1.0,
            lda: in_features as i32,
            ldb: in_features as i32,
            beta: 0.0,
            ldc: out_features as i32,
        };
        unsafe { self.blas.gemm(cfg, weights, input, &mut output) }
            .map_err(|e| format!("Launching cublasSgemm failed: {e:?}"))?;
        self.add_bias(stream, &mut output, bias, batch, out_features)?;
        Ok(output)
    }

    fn gemv_batched(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, String> {
        // cuBLAS has no inefficient parallel-rows mode; the batched-GEMV
        // baseline coincides with GEMM here (bench will show them equal).
        self.gemm(stream, input, weights, bias, batch, in_features, out_features)
    }
}
