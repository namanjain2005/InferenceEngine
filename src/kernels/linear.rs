use super::{LinearOp, LinearWeights, WeightFormat};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

const LINEAR_CUDA: &str = include_str!("layer/linear.cu");
const GEMM_CUDA: &str = include_str!("layer/gemm.cu");

pub struct LinearKernel {
    format: WeightFormat,
    _module: Arc<CudaModule>,
    /// Baseline token-at-a-time GEMV path (kept for decode + comparison).
    function: CudaFunction,
    _gemm_module: Arc<CudaModule>,
    /// Blocked prefill GEMM path (BM=32 x BN=128, TM=4 x TN=4 register tile).
    gemm_function: CudaFunction,
}

impl LinearKernel {
    pub fn new(context: &Arc<CudaContext>) -> Result<Self, String> {
        Self::new_with_format(context, WeightFormat::F32)
    }

    pub fn new_with_format(
        context: &Arc<CudaContext>,
        format: WeightFormat,
    ) -> Result<Self, String> {
        let ptx =
            compile_ptx(LINEAR_CUDA).map_err(|e| format!("Linear CUDA compilation failed: {e}"))?;
        let module = context
            .load_module(ptx)
            .map_err(|e| format!("Loading linear CUDA module failed: {e}"))?;
        let function = module
            .load_function("linear")
            .map_err(|e| format!("Loading linear CUDA function failed: {e}"))?;
        let gemm_ptx =
            compile_ptx(GEMM_CUDA).map_err(|e| format!("GEMM CUDA compilation failed: {e}"))?;
        let gemm_module = context
            .load_module(gemm_ptx)
            .map_err(|e| format!("Loading GEMM CUDA module failed: {e}"))?;
        let gemm_function = gemm_module
            .load_function("gemm")
            .map_err(|e| format!("Loading GEMM CUDA function failed: {e}"))?;
        Ok(Self {
            format,
            _module: module,
            function,
            _gemm_module: gemm_module,
            gemm_function,
        })
    }

    pub fn upload(
        &self,
        stream: &Arc<CudaStream>,
        values: &[f32],
        name: &str,
    ) -> Result<CudaSlice<f32>, String> {
        // `clone_htod` queues an async transfer. The caller often passes a
        // temporary Vec returned by SafeTensors, so keep its memory alive until
        // the transfer has completed.
        let device = stream
            .clone_htod(values)
            .map_err(|e| format!("Uploading {name} to CUDA failed: {e}"))?;
        stream
            .synchronize()
            .map_err(|e| format!("Waiting for {name} upload failed: {e}"))?;
        Ok(device)
    }

    pub fn upload_weight(
        &self,
        stream: &Arc<CudaStream>,
        values: &[f32],
        rows: usize,
        cols: usize,
        name: &str,
    ) -> Result<LinearWeights, String> {
        if rows.checked_mul(cols) != Some(values.len()) {
            return Err(format!(
                "{name}: projection shape {rows}x{cols} does not match {} values",
                values.len()
            ));
        }
        match self.format {
            WeightFormat::F32 => Ok(LinearWeights::F32(self.upload(stream, values, name)?)),
            WeightFormat::F16 => {
                let bits: Vec<u16> = values.iter().map(|&v| f32_to_f16_bits(v)).collect();
                let device = stream
                    .clone_htod(&bits)
                    .map_err(|e| format!("Uploading {name} FP16 weights failed: {e}"))?;
                stream
                    .synchronize()
                    .map_err(|e| format!("Waiting for {name} FP16 upload failed: {e}"))?;
                Ok(LinearWeights::F16(device))
            }
            WeightFormat::I8 => {
                let (quantized, scales) = quantize_rows_i8(values, rows, cols);
                let q_dev = stream
                    .clone_htod(&quantized)
                    .map_err(|e| format!("Uploading {name} INT8 weights failed: {e}"))?;
                let s_dev = stream
                    .clone_htod(&scales)
                    .map_err(|e| format!("Uploading {name} INT8 scales failed: {e}"))?;
                stream
                    .synchronize()
                    .map_err(|e| format!("Waiting for {name} INT8 upload failed: {e}"))?;
                Ok(LinearWeights::I8 {
                    values: q_dev,
                    scales: s_dev,
                })
            }
        }
    }

    pub fn forward_weight(
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
                self.forward(stream, input, w, bias, batch, in_features, out_features)
            }
            LinearWeights::F16(w) => self.forward_low(
                stream,
                input,
                w,
                None,
                bias,
                batch,
                in_features,
                out_features,
                1,
            ),
            LinearWeights::I8 { values, scales } => self.forward_low(
                stream,
                input,
                values,
                Some(scales),
                bias,
                batch,
                in_features,
                out_features,
                2,
            ),
        }
    }

    fn forward_low<T: cudarc::driver::DeviceRepr>(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &CudaSlice<T>,
        scales: Option<&CudaSlice<f32>>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
        in_features: usize,
        out_features: usize,
        dtype: i32,
    ) -> Result<CudaSlice<f32>, String> {
        if batch == 0 || in_features == 0 || out_features == 0 {
            return Err("Linear dimensions must be non-zero".into());
        }
        let input_len = batch
            .checked_mul(in_features)
            .ok_or("Linear input size overflow")?;
        let weight_len = out_features
            .checked_mul(in_features)
            .ok_or("Linear weight size overflow")?;
        if input.len() != input_len
            || weights.len() != weight_len
            || bias.is_some_and(|b| b.len() != out_features)
            || scales.is_some_and(|s| s.len() != out_features)
        {
            return Err("Low-precision linear device shape mismatch".into());
        }
        let zero_bias;
        let bias = match bias {
            Some(b) => b,
            None => {
                zero_bias = stream
                    .alloc_zeros::<f32>(out_features)
                    .map_err(|e| format!("Allocating linear bias failed: {e}"))?;
                &zero_bias
            }
        };
        let zero_scales;
        let scales = match scales {
            Some(s) => s,
            None => {
                zero_scales = stream
                    .alloc_zeros::<f32>(out_features)
                    .map_err(|e| format!("Allocating scales failed: {e}"))?;
                &zero_scales
            }
        };
        let mut output = stream
            .alloc_zeros::<f32>(input_len / in_features * out_features)
            .map_err(|e| format!("Allocating linear output failed: {e}"))?;
        let f = self
            ._module
            .load_function("linear_mixed")
            .map_err(|e| format!("Loading mixed linear kernel failed: {e}"))?;
        let (b, k, n) = (batch as i32, in_features as i32, out_features as i32);
        let mut launch = stream.launch_builder(&f);
        launch.arg(input);
        launch.arg(weights);
        launch.arg(scales);
        launch.arg(bias);
        launch.arg(&mut output);
        launch.arg(&b);
        launch.arg(&k);
        launch.arg(&n);
        launch.arg(&dtype);
        let blocks = batch * out_features.div_ceil(8);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| format!("Launching mixed linear kernel failed: {e}"))?;
        Ok(output)
    }

    pub fn forward_gemm_weight(
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
                self.forward_gemm(stream, input, w, bias, batch, in_features, out_features)
            }
            LinearWeights::F16(w) => self.forward_gemm_low(
                stream,
                input,
                w,
                None,
                bias,
                batch,
                in_features,
                out_features,
                1,
            ),
            LinearWeights::I8 { values, scales } => self.forward_gemm_low(
                stream,
                input,
                values,
                Some(scales),
                bias,
                batch,
                in_features,
                out_features,
                2,
            ),
        }
    }

    fn forward_gemm_low<T: cudarc::driver::DeviceRepr>(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &CudaSlice<T>,
        scales: Option<&CudaSlice<f32>>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
        in_features: usize,
        out_features: usize,
        dtype: i32,
    ) -> Result<CudaSlice<f32>, String> {
        if batch <= 1 {
            return self.forward_low(
                stream,
                input,
                weights,
                scales,
                bias,
                batch.max(1),
                in_features,
                out_features,
                dtype,
            );
        }
        let zero_bias;
        let bias = match bias {
            Some(b) => b,
            None => {
                zero_bias = stream
                    .alloc_zeros::<f32>(out_features)
                    .map_err(|e| format!("Allocating GEMM bias failed: {e}"))?;
                &zero_bias
            }
        };
        let zero_scales;
        let scales = match scales {
            Some(s) => s,
            None => {
                zero_scales = stream
                    .alloc_zeros::<f32>(out_features)
                    .map_err(|e| format!("Allocating GEMM scales failed: {e}"))?;
                &zero_scales
            }
        };
        let mut output = stream
            .alloc_zeros::<f32>(
                batch
                    .checked_mul(out_features)
                    .ok_or("GEMM output size overflow")?,
            )
            .map_err(|e| format!("Allocating GEMM output failed: {e}"))?;
        let f = self
            ._gemm_module
            .load_function("gemm_mixed")
            .map_err(|e| format!("Loading mixed GEMM kernel failed: {e}"))?;
        let (m, n, k) = (batch as i32, out_features as i32, in_features as i32);
        let mut launch = stream.launch_builder(&f);
        launch.arg(input);
        launch.arg(weights);
        launch.arg(scales);
        launch.arg(bias);
        launch.arg(&mut output);
        launch.arg(&m);
        launch.arg(&n);
        launch.arg(&k);
        launch.arg(&dtype);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (
                    out_features.div_ceil(128) as u32,
                    batch.div_ceil(32) as u32,
                    1,
                ),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| format!("Launching mixed GEMM kernel failed: {e}"))?;
        Ok(output)
    }

    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, String> {
        if batch == 0 || in_features == 0 || out_features == 0 {
            return Err("Linear dimensions must be non-zero".into());
        }
        let input_len = batch
            .checked_mul(in_features)
            .ok_or("Linear input size overflow")?;
        let weight_len = out_features
            .checked_mul(in_features)
            .ok_or("Linear weight size overflow")?;
        let output_len = batch
            .checked_mul(out_features)
            .ok_or("Linear output size overflow")?;
        if input.len() != input_len
            || weights.len() != weight_len
            || bias.map_or(false, |b| b.len() != out_features)
        {
            return Err(format!(
                "Linear device shape mismatch: input {}, weights {}, bias {:?}; expected {input_len}, {weight_len}, {out_features}",
                input.len(),
                weights.len(),
                bias.map(|b| b.len())
            ));
        }
        if [batch, in_features, out_features, output_len]
            .iter()
            .any(|&v| v > i32::MAX as usize)
        {
            return Err("Linear dimensions exceed CUDA kernel index range".into());
        }
        let bias_zeros;
        let bias = match bias {
            Some(b) => b,
            None => {
                bias_zeros = stream
                    .alloc_zeros::<f32>(out_features)
                    .map_err(|e| format!("Allocating linear zero bias failed: {e}"))?;
                &bias_zeros
            }
        };
        let mut output = stream
            .alloc_zeros::<f32>(output_len)
            .map_err(|e| format!("Allocating linear output failed: {e}"))?;
        let (batch_i32, input_n, output_n) =
            (batch as i32, in_features as i32, out_features as i32);
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(input);
        launch.arg(weights);
        launch.arg(bias);
        launch.arg(&mut output);
        launch.arg(&batch_i32);
        launch.arg(&input_n);
        launch.arg(&output_n);
        let threads = 256u32;
        let outputs_per_block = threads as usize / 32;
        let blocks_per_row = out_features.div_ceil(outputs_per_block);
        let grid_x = batch
            .checked_mul(blocks_per_row)
            .ok_or("Linear grid size overflow")?;
        let grid_x = u32::try_from(grid_x).map_err(|_| "Linear grid exceeds CUDA launch range")?;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(cfg) }
            .map_err(|e| format!("Launching linear CUDA kernel failed: {e}"))?;
        Ok(output)
    }

    /// Blocked GEMM prefill path sharing the same weight slices as `forward`.
    /// Computes C[M,N] = A[M,K] * W[N,K]^T + bias. Routes M==1 to the GEMV
    /// path since blocking has no reuse across a single row.
    pub fn forward_gemm(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaSlice<f32>,
        weights: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<CudaSlice<f32>, String> {
        if batch <= 1 {
            return self.forward(
                stream,
                input,
                weights,
                bias,
                batch.max(1),
                in_features,
                out_features,
            );
        }
        if in_features == 0 || out_features == 0 {
            return Err("Linear dimensions must be non-zero".into());
        }
        let input_len = batch
            .checked_mul(in_features)
            .ok_or("Linear input size overflow")?;
        let weight_len = out_features
            .checked_mul(in_features)
            .ok_or("Linear weight size overflow")?;
        let output_len = batch
            .checked_mul(out_features)
            .ok_or("Linear output size overflow")?;
        if input.len() != input_len
            || weights.len() != weight_len
            || bias.map_or(false, |b| b.len() != out_features)
        {
            return Err(format!(
                "GEMM device shape mismatch: input {}, weights {}, bias {:?}; expected {input_len}, {weight_len}, {out_features}",
                input.len(),
                weights.len(),
                bias.map(|b| b.len())
            ));
        }
        if [batch, in_features, out_features, output_len]
            .iter()
            .any(|&v| v > i32::MAX as usize)
        {
            return Err("Linear dimensions exceed CUDA kernel index range".into());
        }
        let bias_zeros;
        let bias = match bias {
            Some(b) => b,
            // Bias is read once per output column; keep the same zero-fill
            // convention as the GEMV path so logits match exactly.
            None => {
                bias_zeros = stream
                    .alloc_zeros::<f32>(out_features)
                    .map_err(|e| format!("Allocating GEMM zero bias failed: {e}"))?;
                &bias_zeros
            }
        };
        let mut output = stream
            .alloc_zeros::<f32>(output_len)
            .map_err(|e| format!("Allocating GEMM output failed: {e}"))?;
        let (m, n, k) = (batch as i32, out_features as i32, in_features as i32);
        let mut launch = stream.launch_builder(&self.gemm_function);
        launch.arg(input);
        launch.arg(weights);
        launch.arg(bias);
        launch.arg(&mut output);
        launch.arg(&m);
        launch.arg(&n);
        launch.arg(&k);
        // Must match BM/BN in gemm.cu.
        let grid_x = out_features.div_ceil(128) as u32;
        let grid_y = batch.div_ceil(32) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(cfg) }
            .map_err(|e| format!("Launching GEMM CUDA kernel failed: {e}"))?;
        Ok(output)
    }
}

impl LinearOp for LinearKernel {
    fn upload(
        &self,
        stream: &Arc<CudaStream>,
        values: &[f32],
        name: &str,
    ) -> Result<CudaSlice<f32>, String> {
        LinearKernel::upload(self, stream, values, name)
    }

    fn upload_weight(
        &self,
        stream: &Arc<CudaStream>,
        values: &[f32],
        rows: usize,
        cols: usize,
        name: &str,
    ) -> Result<LinearWeights, String> {
        LinearKernel::upload_weight(self, stream, values, rows, cols, name)
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
        self.forward(stream, input, weights, bias, 1, in_features, out_features)
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
        self.forward_gemm(
            stream,
            input,
            weights,
            bias,
            batch,
            in_features,
            out_features,
        )
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
        self.forward(
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
        self.forward_weight(stream, input, weights, bias, 1, in_features, out_features)
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
        self.forward_gemm_weight(
            stream,
            input,
            weights,
            bias,
            batch,
            in_features,
            out_features,
        )
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
        self.forward_weight(
            stream,
            input,
            weights,
            bias,
            batch,
            in_features,
            out_features,
        )
    }
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7fffff;
    if exp == 255 {
        return sign | if frac == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = frac | 0x800000;
        let shift = (14 - half_exp) as u32;
        let rounded = (mantissa + (1 << (shift - 1)) - 1 + ((mantissa >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    let rounded = frac + 0xfff + ((frac >> 13) & 1);
    let e = if rounded & 0x800000 != 0 {
        half_exp + 1
    } else {
        half_exp
    };
    if e >= 31 {
        sign | 0x7c00
    } else {
        sign | ((e as u16) << 10) | (((rounded >> 13) & 0x3ff) as u16)
    }
}

fn quantize_rows_i8(values: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut quantized = vec![0i8; values.len()];
    let mut scales = vec![1.0f32; rows];
    for r in 0..rows {
        let row = &values[r * cols..(r + 1) * cols];
        let max_abs = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        scales[r] = scale;
        for (q, &v) in quantized[r * cols..(r + 1) * cols].iter_mut().zip(row) {
            *q = (v / scale).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (quantized, scales)
}

#[cfg(test)]
mod precision_tests {
    use super::{f32_to_f16_bits, quantize_rows_i8};

    #[test]
    fn f32_to_f16_encodes_normal_subnormal_and_special_values() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(2.0f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(f32_to_f16_bits(f32::NAN) & 0x7c00, 0x7c00);
    }

    #[test]
    fn int8_quantization_uses_independent_row_scales_and_handles_zero_rows() {
        let (q, scales) = quantize_rows_i8(&[0.0, -1.0, 0.5, 0.0, 0.0, 0.0], 2, 3);
        assert_eq!(&q[..3], &[0, -127, 64]);
        assert_eq!(q[3..], [0, 0, 0]);
        assert!((scales[0] - 1.0 / 127.0).abs() < 1e-8);
        assert_eq!(scales[1], 1.0);
    }
}
