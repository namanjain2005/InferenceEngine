use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

const LINEAR_CUDA: &str = include_str!("layer/linear.cu");

pub struct LinearKernel {
    _module: Arc<CudaModule>,
    function: CudaFunction,
}

impl LinearKernel {
    pub fn new(context: &Arc<CudaContext>) -> Result<Self, String> {
        let ptx = compile_ptx(LINEAR_CUDA).map_err(|e| format!("Linear CUDA compilation failed: {e}"))?;
        let module = context.load_module(ptx).map_err(|e| format!("Loading linear CUDA module failed: {e}"))?;
        let function = module.load_function("linear").map_err(|e| format!("Loading linear CUDA function failed: {e}"))?;
        Ok(Self { _module: module, function })
    }

    pub fn upload(&self, stream: &Arc<CudaStream>, values: &[f32], name: &str) -> Result<CudaSlice<f32>, String> {
        // `clone_htod` queues an async transfer. The caller often passes a
        // temporary Vec returned by SafeTensors, so keep its memory alive until
        // the transfer has completed.
        let device = stream.clone_htod(values).map_err(|e| format!("Uploading {name} to CUDA failed: {e}"))?;
        stream.synchronize().map_err(|e| format!("Waiting for {name} upload failed: {e}"))?;
        Ok(device)
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
        if batch == 0 || in_features == 0 || out_features == 0 { return Err("Linear dimensions must be non-zero".into()); }
        let input_len = batch.checked_mul(in_features).ok_or("Linear input size overflow")?;
        let weight_len = out_features.checked_mul(in_features).ok_or("Linear weight size overflow")?;
        let output_len = batch.checked_mul(out_features).ok_or("Linear output size overflow")?;
        if input.len() != input_len || weights.len() != weight_len || bias.map_or(false, |b| b.len() != out_features) {
            return Err(format!("Linear device shape mismatch: input {}, weights {}, bias {:?}; expected {input_len}, {weight_len}, {out_features}", input.len(), weights.len(), bias.map(|b| b.len())));
        }
        if [batch, in_features, out_features, output_len].iter().any(|&v| v > i32::MAX as usize) {
            return Err("Linear dimensions exceed CUDA kernel index range".into());
        }
        let bias_zeros;
        let bias = match bias {
            Some(b) => b,
            None => { bias_zeros = stream.alloc_zeros::<f32>(out_features).map_err(|e| format!("Allocating linear zero bias failed: {e}"))?; &bias_zeros }
        };
        let mut output = stream.alloc_zeros::<f32>(output_len).map_err(|e| format!("Allocating linear output failed: {e}"))?;
        let (batch_i32, input_n, output_n) = (batch as i32, in_features as i32, out_features as i32);
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(input); launch.arg(weights); launch.arg(bias); launch.arg(&mut output);
        launch.arg(&batch_i32); launch.arg(&input_n); launch.arg(&output_n);
        let threads = 256u32;
        let outputs_per_block = threads as usize / 32;
        let blocks_per_row = out_features.div_ceil(outputs_per_block);
        let grid_x = batch.checked_mul(blocks_per_row).ok_or("Linear grid size overflow")?;
        let grid_x = u32::try_from(grid_x).map_err(|_| "Linear grid exceeds CUDA launch range")?;
        let cfg = LaunchConfig { grid_dim: (grid_x, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe { launch.launch(cfg) }.map_err(|e| format!("Launching linear CUDA kernel failed: {e}"))?;
        Ok(output)
    }
}
