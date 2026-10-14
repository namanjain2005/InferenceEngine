pub mod cublas_linear;
pub mod linear;
pub mod ops;

use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

/// Backend contract for the linear (fully-connected) projections. Both the
/// hand-rolled kernels (`linear::LinearKernel`) and cuBLAS
/// (`cublas_linear::CublasLinear`) implement it over the same row-major
/// layout: input `[batch, K]`, weights `[N, K]`, output `[batch, N]` with
/// `out[m, n] = sum_k input[m, k] * weights[n, k] + bias[n]`.
///
/// Kept side by side so later work can pick per-case implementations
/// (e.g. fused custom kernels) behind one call shape.
pub trait LinearOp {
    fn upload(&self, stream: &Arc<CudaStream>, values: &[f32], name: &str) -> Result<CudaSlice<f32>, String>;

    /// Single-token path (decode). Input has `in_features` elements;
    /// output has `out_features`.
    fn gemv(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, String>;

    /// Batched path (prefill). Input/output are `[batch, K]` / `[batch, N]`.
    fn gemm(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, String>;

    /// Batched path with GEMV-style parallel-rows math (custom baseline).
    /// Backends without a distinct row mode may route this to `gemm`.
    fn gemv_batched(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, String>;
}
