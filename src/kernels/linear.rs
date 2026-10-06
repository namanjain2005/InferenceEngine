use cudarc::driver::{CudaContext, CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

const LINEAR_CUDA: &str = include_str!("layer/linear.cu");

pub struct LinearKernel {
    _module: Arc<CudaModule>,
    function: CudaFunction,
}

impl LinearKernel {
    pub fn new(context: &Arc<CudaContext>) -> Result<Self, String> {
        let ptx = compile_ptx(LINEAR_CUDA)
            .map_err(|e| format!("Linear CUDA compilation failed: {e}"))?;
        let module = context
            .load_module(ptx)
            .map_err(|e| format!("Loading linear CUDA module failed: {e}"))?;
        let function = module
            .load_function("linear")
            .map_err(|e| format!("Loading linear CUDA function failed: {e}"))?;
        Ok(Self { _module: module, function })
    }

    /// Computes Y = X * W^T + bias for row-major X [batch, in_features]
    /// and W [out_features, in_features]. An empty bias slice disables bias.
    pub fn forward(
        &self,
        stream: &Arc<cudarc::driver::CudaStream>,
        input: &[f32],
        weights: &[f32],
        bias: Option<&[f32]>,
        batch: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Vec<f32>, String> {
        if batch == 0 || in_features == 0 || out_features == 0 {
            return Err("Linear dimensions must be non-zero".into());
        }
        let input_len = batch.checked_mul(in_features).ok_or("Linear input size overflow")?;
        let weight_len = out_features.checked_mul(in_features).ok_or("Linear weight size overflow")?;
        let output_len = batch.checked_mul(out_features).ok_or("Linear output size overflow")?;
        if batch > i32::MAX as usize || in_features > i32::MAX as usize || out_features > i32::MAX as usize
            || output_len > i32::MAX as usize
        {
            return Err("Linear dimensions exceed the CUDA kernel index range".into());
        }
        if input.len() != input_len || weights.len() != weight_len {
            return Err(format!(
                "Linear shape mismatch: input has {}, expected {}; weights have {}, expected {}",
                input.len(), input_len, weights.len(), weight_len
            ));
        }
        if let Some(b) = bias {
            if b.len() != out_features {
                return Err(format!("Linear bias has {}, expected {out_features}", b.len()));
            }
        }

        let input_dev = stream.clone_htod(input).map_err(|e| format!("Copying linear input failed: {e}"))?;
        let weights_dev = stream.clone_htod(weights).map_err(|e| format!("Copying linear weights failed: {e}"))?;
        let bias_host = bias.map_or_else(|| vec![0.0; out_features], ToOwned::to_owned);
        let bias_dev = stream.clone_htod(&bias_host).map_err(|e| format!("Copying linear bias failed: {e}"))?;
        let mut output_dev = stream.alloc_zeros::<f32>(output_len)
            .map_err(|e| format!("Allocating linear output failed: {e}"))?;

        let batch_i32 = i32::try_from(batch).map_err(|_| "Linear batch exceeds CUDA index range")?;
        let in_i32 = i32::try_from(in_features).map_err(|_| "Linear input size exceeds CUDA index range")?;
        let out_i32 = i32::try_from(out_features).map_err(|_| "Linear output size exceeds CUDA index range")?;
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(&input_dev);
        launch.arg(&weights_dev);
        launch.arg(&bias_dev);
        launch.arg(&mut output_dev);
        launch.arg(&batch_i32);
        launch.arg(&in_i32);
        launch.arg(&out_i32);
        let threads = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (output_len.div_ceil(threads as usize) as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(cfg) }.map_err(|e| format!("Launching linear CUDA kernel failed: {e}"))?;
        stream.clone_dtoh(&output_dev).map_err(|e| format!("Copying linear output failed: {e}"))
    }
}
