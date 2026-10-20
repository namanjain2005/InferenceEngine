pub mod cublas_linear;
pub mod linear;
pub mod ops;

use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

/// Projection weights kept in their resident storage format. Activations and
/// accumulation remain f32; F16 and symmetric per-row I8 reduce weight VRAM.
pub enum LinearWeights {
    F32(CudaSlice<f32>),
    F16(CudaSlice<u16>),
    I8 {
        values: CudaSlice<i8>,
        scales: CudaSlice<f32>,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WeightFormat {
    #[default]
    F32,
    F16,
    I8,
}

/// Backend contract for the linear (fully-connected) projections. Both the
/// hand-rolled kernels (`linear::LinearKernel`) and cuBLAS
/// (`cublas_linear::CublasLinear`) implement it over the same row-major
/// layout: input `[batch, K]`, weights `[N, K]`, output `[batch, N]` with
/// `out[m, n] = sum_k input[m, k] * weights[n, k] + bias[n]`.
///
/// Kept side by side so later work can pick per-case implementations
/// (e.g. fused custom kernels) behind one call shape.
pub trait LinearOp {
    fn upload(
        &self,
        stream: &Arc<CudaStream>,
        values: &[f32],
        name: &str,
    ) -> Result<CudaSlice<f32>, String>;
    fn upload_weight(
        &self,
        stream: &Arc<CudaStream>,
        values: &[f32],
        rows: usize,
        cols: usize,
        name: &str,
    ) -> Result<LinearWeights, String>;

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

    fn gemv_weight(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &LinearWeights,
        bias: Option<&CudaSlice<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, String>;
    fn gemm_weight(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &LinearWeights,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, String>;
    fn gemv_batched_weight(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &LinearWeights,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, String>;
}
