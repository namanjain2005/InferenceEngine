use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

const RMS_NORM_CUDA: &str = include_str!("RMSNorm.cu");
const SWIGLU_CUDA: &str = include_str!("SWiGLU.cu");
const ATTENTION_CUDA: &str = include_str!("../layer/attention.cu");
const THREADS: u32 = 256;

fn load_kernel(
    context: &Arc<CudaContext>,
    source: &str,
    function_name: &str,
) -> Result<(Arc<CudaModule>, CudaFunction), String> {
    let ptx = compile_ptx(source)
        .map_err(|e| format!("Compiling {function_name} CUDA kernel failed: {e}"))?;
    let module = context
        .load_module(ptx)
        .map_err(|e| format!("Loading {function_name} CUDA module failed: {e}"))?;
    let function = module
        .load_function(function_name)
        .map_err(|e| format!("Loading {function_name} CUDA function failed: {e}"))?;
    Ok((module, function))
}

fn checked_i32(value: usize, name: &str) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("{name} exceeds CUDA kernel index range"))
}

pub struct RmsNormKernel {
    _module: Arc<CudaModule>,
    function: CudaFunction,
}

impl RmsNormKernel {
    pub fn new(context: &Arc<CudaContext>) -> Result<Self, String> {
        let (module, function) = load_kernel(context, RMS_NORM_CUDA, "rmsnorm")?;
        Ok(Self { _module: module, function })
    }

    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        input: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, String> {
        if input.is_empty() || input.len() != weight.len() {
            return Err(format!("RMSNorm shape mismatch: input has {}, weight has {}", input.len(), weight.len()));
        }
        checked_i32(input.len(), "RMSNorm width")?;
        let input_dev = stream.clone_htod(input).map_err(|e| format!("Copying RMSNorm input failed: {e}"))?;
        let weight_dev = stream.clone_htod(weight).map_err(|e| format!("Copying RMSNorm weight failed: {e}"))?;
        let mut output_dev = stream.alloc_zeros::<f32>(input.len()).map_err(|e| format!("Allocating RMSNorm output failed: {e}"))?;
        let n = input.len() as i32;
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(&input_dev);
        launch.arg(&weight_dev);
        launch.arg(&mut output_dev);
        launch.arg(&n);
        launch.arg(&eps);
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: THREADS * 4 };
        unsafe { launch.launch(cfg) }.map_err(|e| format!("Launching RMSNorm CUDA kernel failed: {e}"))?;
        stream.clone_dtoh(&output_dev).map_err(|e| format!("Copying RMSNorm output failed: {e}"))
    }
}

pub struct SwiGluKernel {
    _module: Arc<CudaModule>,
    function: CudaFunction,
}

impl SwiGluKernel {
    pub fn new(context: &Arc<CudaContext>) -> Result<Self, String> {
        let (module, function) = load_kernel(context, SWIGLU_CUDA, "SWiGLU")?;
        Ok(Self { _module: module, function })
    }

    /// Computes `silu(gate[i]) * up[i]` from concatenated `[gate, up]` input.
    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        gate: &[f32],
        up: &[f32],
    ) -> Result<Vec<f32>, String> {
        if gate.is_empty() || gate.len() != up.len() {
            return Err(format!("SwiGLU shape mismatch: gate has {}, up has {}", gate.len(), up.len()));
        }
        let half_n = checked_i32(gate.len(), "SwiGLU width")?;
        let total = gate.len().checked_mul(2).ok_or("SwiGLU input size overflow")?;
        let total_i32 = checked_i32(total, "SwiGLU input size")?;
        let mut packed = Vec::with_capacity(total);
        packed.extend_from_slice(gate);
        packed.extend_from_slice(up);
        let input_dev = stream.clone_htod(&packed).map_err(|e| format!("Copying SwiGLU input failed: {e}"))?;
        let mut output_dev = stream.alloc_zeros::<f32>(gate.len()).map_err(|e| format!("Allocating SwiGLU output failed: {e}"))?;
        let batch = 1i32;
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(&input_dev);
        launch.arg(&mut output_dev);
        launch.arg(&batch);
        launch.arg(&half_n);
        let grid = ((total_i32 + THREADS as i32 - 1) / THREADS as i32) as u32;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 };
        unsafe { launch.launch(cfg) }.map_err(|e| format!("Launching SwiGLU CUDA kernel failed: {e}"))?;
        stream.clone_dtoh(&output_dev).map_err(|e| format!("Copying SwiGLU output failed: {e}"))
    }
}

pub struct AttentionKernel {
    _module: Arc<CudaModule>,
    function: CudaFunction,
}

impl AttentionKernel {
    pub fn new(context: &Arc<CudaContext>) -> Result<Self, String> {
        let (module, function) = load_kernel(context, ATTENTION_CUDA, "attention_decode")?;
        Ok(Self { _module: module, function })
    }

    pub fn forward(
        &self,
        stream: &Arc<CudaStream>,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>, String> {
        if seq_len == 0 || num_q_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
            return Err("Attention dimensions must be non-zero".into());
        }
        if num_q_heads % num_kv_heads != 0 {
            return Err("Attention query head count must be divisible by key/value head count".into());
        }
        let kv_dim = num_kv_heads.checked_mul(head_dim).ok_or("Attention KV dimension overflow")?;
        let q_len = num_q_heads.checked_mul(head_dim).ok_or("Attention query size overflow")?;
        let cache_len = seq_len.checked_mul(kv_dim).ok_or("Attention cache size overflow")?;
        if q.len() != q_len || k_cache.len() != cache_len || v_cache.len() != cache_len {
            return Err(format!("Attention shape mismatch: q {}, k {}, v {}; expected {}, {}, {}", q.len(), k_cache.len(), v_cache.len(), q_len, cache_len, cache_len));
        }
        let seq_i32 = checked_i32(seq_len, "Attention sequence length")?;
        let kv_i32 = checked_i32(kv_dim, "Attention KV dimension")?;
        let head_i32 = checked_i32(head_dim, "Attention head dimension")?;
        let heads_per_kv = checked_i32(num_q_heads / num_kv_heads, "Attention heads per KV head")?;
        checked_i32(num_q_heads, "Attention query head count")?;
        checked_i32(seq_len.checked_mul(4).ok_or("Attention shared memory size overflow")?, "Attention shared memory size")?;

        let q_dev = stream.clone_htod(q).map_err(|e| format!("Copying attention query failed: {e}"))?;
        let k_dev = stream.clone_htod(k_cache).map_err(|e| format!("Copying attention key cache failed: {e}"))?;
        let v_dev = stream.clone_htod(v_cache).map_err(|e| format!("Copying attention value cache failed: {e}"))?;
        let mut output_dev = stream.alloc_zeros::<f32>(q_len).map_err(|e| format!("Allocating attention output failed: {e}"))?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(&q_dev);
        launch.arg(&k_dev);
        launch.arg(&v_dev);
        launch.arg(&mut output_dev);
        launch.arg(&seq_i32);
        launch.arg(&kv_i32);
        launch.arg(&head_i32);
        launch.arg(&heads_per_kv);
        launch.arg(&scale);
        let cfg = LaunchConfig { grid_dim: (num_q_heads as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: (seq_len * 4) as u32 };
        unsafe { launch.launch(cfg) }.map_err(|e| format!("Launching attention CUDA kernel failed: {e}"))?;
        stream.clone_dtoh(&output_dev).map_err(|e| format!("Copying attention output failed: {e}"))
    }
}
