use super::linear::LinearKernel;
use super::{LinearOp, LinearWeights, WeightFormat};
use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig, Gemv, GemvConfig};
use cudarc::cublas::sys;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
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
    reduced: LinearKernel,
    blas: CudaBlas,
    _bias_module: Arc<CudaModule>,
    bias_fn: CudaFunction,
    cast_fn: CudaFunction,
    dequant_fn: CudaFunction,
}

impl CublasLinear {
    pub fn new(context: &Arc<CudaContext>, stream: Arc<CudaStream>) -> Result<Self, String> {
        Self::new_with_format(context, stream, WeightFormat::F32)
    }

    pub fn new_with_format(
        context: &Arc<CudaContext>,
        stream: Arc<CudaStream>,
        format: WeightFormat,
    ) -> Result<Self, String> {
        let blas =
            CudaBlas::new(stream).map_err(|e| format!("Creating cuBLAS handle failed: {e:?}"))?;
        let reduced = LinearKernel::new_with_format(context, format)?;
        let ptx =
            compile_ptx(BIAS_CUDA).map_err(|e| format!("Bias CUDA compilation failed: {e}"))?;
        let module = context
            .load_module(ptx)
            .map_err(|e| format!("Loading bias CUDA module failed: {e}"))?;
        let bias_fn = module
            .load_function("bias_add")
            .map_err(|e| format!("Loading bias_add function failed: {e}"))?;
        let cast_fn = module
            .load_function("cast_f32_to_f16")
            .map_err(|e| format!("Loading f32-to-f16 cast function failed: {e}"))?;
        let dequant_fn = module
            .load_function("dequant_i8_rows")
            .map_err(|e| format!("Loading INT8 dequant function failed: {e}"))?;
        Ok(Self {
            reduced,
            blas,
            _bias_module: module,
            bias_fn,
            cast_fn,
            dequant_fn,
        })
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
        let input_len_exp = batch
            .checked_mul(in_features)
            .ok_or("cuBLAS linear input size overflow")?;
        let weight_len_exp = out_features
            .checked_mul(in_features)
            .ok_or("cuBLAS linear weight size overflow")?;
        if input_len != input_len_exp
            || weight_len != weight_len_exp
            || bias_len.map_or(false, |b| b != out_features)
        {
            return Err(format!(
                "cuBLAS linear device shape mismatch: input {input_len}, weights {weight_len}, bias {bias_len:?}; expected {input_len_exp}, {weight_len_exp}, {out_features}"
            ));
        }
        if [batch, in_features, out_features]
            .iter()
            .any(|&v| v > i32::MAX as usize)
        {
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
        let total = batch
            .checked_mul(out_features)
            .ok_or("cuBLAS bias size overflow")?;
        let grid = total.div_ceil(256) as u32;
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| format!("Launching bias_add kernel failed: {e}"))?;
        Ok(())
    }

    fn gemm_f16_weights(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &CudaSlice<u16>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, String> {
        Self::check(
            input.len(),
            weights.len(),
            bias.map(|b| b.len()),
            batch,
            in_features,
            out_features,
        )?;
        let mut output = stream
            .alloc_zeros::<f32>(batch * out_features)
            .map_err(|e| format!("Allocating FP16-weight GEMM output failed: {e}"))?;
        let input_len = batch
            .checked_mul(in_features)
            .ok_or("FP16 input size overflow")?;
        let mut input_half = stream
            .alloc_zeros::<u16>(input_len)
            .map_err(|e| format!("Allocating FP16 input failed: {e}"))?;
        let input_len_i32 = input_len as i32;
        let mut cast = stream.launch_builder(&self.cast_fn);
        cast.arg(input);
        cast.arg(&mut input_half);
        cast.arg(&input_len_i32);
        unsafe {
            cast.launch(LaunchConfig {
                grid_dim: (input_len.div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| format!("Casting FP32 activations to FP16 failed: {e}"))?;
        let (wp, _wg) = weights.device_ptr(stream);
        let (xp, _xg) = input_half.device_ptr(stream);
        let (yp, _yg) = output.device_ptr_mut(stream);
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *self.blas.handle(),
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                out_features as i32,
                batch as i32,
                in_features as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                wp as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_16F,
                in_features as i32,
                xp as *const std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_16F,
                in_features as i32,
                &beta as *const f32 as *const std::ffi::c_void,
                yp as *mut std::ffi::c_void,
                sys::cudaDataType_t::CUDA_R_32F,
                out_features as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }
        .map_err(|e| format!("Launching cuBLAS mixed FP16 GEMM failed: {e:?}"))?;
        drop((_wg, _xg, _yg));
        self.add_bias(stream, &mut output, bias, batch, out_features)?;
        Ok(output)
    }

    fn dequant_i8(
        &self,
        stream: &Arc<CudaStream>,
        values: &CudaSlice<i8>,
        scales: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<CudaSlice<f32>, String> {
        if values.len() != rows.checked_mul(cols).ok_or("INT8 weight size overflow")?
            || scales.len() != rows
        {
            return Err("INT8 weight shape mismatch".into());
        }
        let mut output = stream
            .alloc_zeros::<f32>(values.len())
            .map_err(|e| format!("Allocating dequantized weights failed: {e}"))?;
        let (rows_i32, cols_i32) = (rows as i32, cols as i32);
        let mut launch = stream.launch_builder(&self.dequant_fn);
        launch.arg(values);
        launch.arg(scales);
        launch.arg(&mut output);
        launch.arg(&rows_i32);
        launch.arg(&cols_i32);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (values.len().div_ceil(256) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| format!("Dequantizing INT8 weights failed: {e}"))?;
        Ok(output)
    }
}

impl LinearOp for CublasLinear {
    fn upload(
        &self,
        stream: &Arc<CudaStream>,
        values: &[f32],
        name: &str,
    ) -> Result<CudaSlice<f32>, String> {
        let device = stream
            .clone_htod(values)
            .map_err(|e| format!("Uploading {name} to CUDA failed: {e}"))?;
        stream
            .synchronize()
            .map_err(|e| format!("Waiting for {name} upload failed: {e}"))?;
        Ok(device)
    }

    fn upload_weight(
        &self,
        stream: &Arc<CudaStream>,
        values: &[f32],
        rows: usize,
        cols: usize,
        name: &str,
    ) -> Result<LinearWeights, String> {
        self.reduced.upload_weight(stream, values, rows, cols, name)
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
        Self::check(
            input.len(),
            weights.len(),
            bias.map(|b| b.len()),
            1,
            in_features,
            out_features,
        )?;
        let mut output = stream
            .alloc_zeros::<f32>(out_features)
            .map_err(|e| format!("Allocating cuBLAS gemv output failed: {e}"))?;

        // The row-major [out_features, in_features] weight buffer is the
        // column-major [in_features, out_features] matrix A. Therefore
        // A^T * input produces the desired output vector. cuBLAS GEMV's
        // output length for OP_T is n, so allocate exactly out_features.
        let cfg = GemvConfig {
            trans: sys::cublasOperation_t::CUBLAS_OP_T,
            m: in_features as i32,
            n: out_features as i32,
            alpha: 1.0,
            lda: in_features as i32,
            incx: 1,
            beta: 0.0,
            incy: 1,
        };
        unsafe { self.blas.gemv(cfg, weights, input, &mut output) }
            .map_err(|e| format!("Launching cublasSgemv failed: {e:?}"))?;
        self.add_bias(stream, &mut output, bias, 1, out_features)?;
        Ok(output)
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
        Self::check(
            input.len(),
            weights.len(),
            bias.map(|b| b.len()),
            batch,
            in_features,
            out_features,
        )?;
        let output_len = batch * out_features;
        let mut output = stream
            .alloc_zeros::<f32>(output_len)
            .map_err(|e| format!("Allocating cuBLAS gemm output failed: {e}"))?;
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
        self.gemm(
            stream,
            input,
            weights,
            bias,
            batch,
            in_features,
            out_features,
        )
    }

    fn gemv_weight(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &LinearWeights,
        bias: Option<&CudaSlice<f32>>,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, String> {
        match weights {
            LinearWeights::F32(w) => self.gemv(stream, input, w, bias, in_features, out_features),
            LinearWeights::F16(w) => {
                self.gemm_f16_weights(stream, input, w, bias, 1, in_features, out_features)
            }
            LinearWeights::I8 { values, scales } => {
                let expanded =
                    self.dequant_i8(stream, values, scales, out_features, in_features)?;
                self.gemv(stream, input, &expanded, bias, in_features, out_features)
            }
        }
    }
    fn gemm_weight(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &LinearWeights,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, String> {
        match weights {
            LinearWeights::F32(w) => {
                self.gemm(stream, input, w, bias, batch, in_features, out_features)
            }
            LinearWeights::F16(w) => {
                self.gemm_f16_weights(stream, input, w, bias, batch, in_features, out_features)
            }
            LinearWeights::I8 { values, scales } => {
                let expanded =
                    self.dequant_i8(stream, values, scales, out_features, in_features)?;
                self.gemm(
                    stream,
                    input,
                    &expanded,
                    bias,
                    batch,
                    in_features,
                    out_features,
                )
            }
        }
    }
    fn gemv_batched_weight(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &LinearWeights,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, String> {
        match weights {
            LinearWeights::F32(w) => {
                self.gemv_batched(stream, input, w, bias, batch, in_features, out_features)
            }
            LinearWeights::F16(w) => {
                self.gemm_f16_weights(stream, input, w, bias, batch, in_features, out_features)
            }
            LinearWeights::I8 { values, scales } => {
                let expanded =
                    self.dequant_i8(stream, values, scales, out_features, in_features)?;
                self.gemv_batched(
                    stream,
                    input,
                    &expanded,
                    bias,
                    batch,
                    in_features,
                    out_features,
                )
            }
        }
    }
}
