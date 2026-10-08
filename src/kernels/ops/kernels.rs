use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

const RMS_NORM_CUDA: &str = include_str!("RMSNorm.cu");
const SWIGLU_CUDA: &str = include_str!("SWiGLU.cu");
const ATTENTION_CUDA: &str = include_str!("../layer/attention.cu");
const VECTOR_CUDA: &str = include_str!("vector.cu");
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
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        eps: f32,
    ) -> Result<CudaSlice<f32>, String> {
        if input.is_empty() || input.len() != weight.len() {
            return Err(format!("RMSNorm shape mismatch: input has {}, weight has {}", input.len(), weight.len()));
        }
        checked_i32(input.len(), "RMSNorm width")?;
        let mut output_dev = stream.alloc_zeros::<f32>(input.len()).map_err(|e| format!("Allocating RMSNorm output failed: {e}"))?;
        let n = input.len() as i32;
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(input);
        launch.arg(weight);
        launch.arg(&mut output_dev);
        launch.arg(&n);
        launch.arg(&eps);
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: THREADS * 4 };
        unsafe { launch.launch(cfg) }.map_err(|e| format!("Launching RMSNorm CUDA kernel failed: {e}"))?;
        Ok(output_dev)
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
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, String> {
        if gate.is_empty() || gate.len() != up.len() {
            return Err(format!("SwiGLU shape mismatch: gate has {}, up has {}", gate.len(), up.len()));
        }
        let half_n = checked_i32(gate.len(), "SwiGLU width")?;
        let total_i32 = half_n;
        let mut output_dev = stream.alloc_zeros::<f32>(gate.len()).map_err(|e| format!("Allocating SwiGLU output failed: {e}"))?;
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(gate);
        launch.arg(up);
        launch.arg(&mut output_dev);
        launch.arg(&half_n);
        let grid = ((total_i32 + THREADS as i32 - 1) / THREADS as i32) as u32;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: 0 };
        unsafe { launch.launch(cfg) }.map_err(|e| format!("Launching SwiGLU CUDA kernel failed: {e}"))?;
        Ok(output_dev)
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
        q: &CudaSlice<f32>,
        k_cache: &CudaSlice<f32>,
        v_cache: &CudaSlice<f32>,
        seq_len: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<CudaSlice<f32>, String> {
        if seq_len == 0 || num_q_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
            return Err("Attention dimensions must be non-zero".into());
        }
        if num_q_heads % num_kv_heads != 0 {
            return Err("Attention query head count must be divisible by key/value head count".into());
        }
        let kv_dim = num_kv_heads.checked_mul(head_dim).ok_or("Attention KV dimension overflow")?;
        let q_len = num_q_heads.checked_mul(head_dim).ok_or("Attention query size overflow")?;
        let cache_len = seq_len.checked_mul(kv_dim).ok_or("Attention cache size overflow")?;
        if q.len() != q_len || k_cache.len() < cache_len || v_cache.len() < cache_len {
            return Err(format!("Attention shape mismatch: q {}, k {}, v {}; expected {}, {}, {}", q.len(), k_cache.len(), v_cache.len(), q_len, cache_len, cache_len));
        }
        let seq_i32 = checked_i32(seq_len, "Attention sequence length")?;
        let kv_i32 = checked_i32(kv_dim, "Attention KV dimension")?;
        let head_i32 = checked_i32(head_dim, "Attention head dimension")?;
        let heads_per_kv = checked_i32(num_q_heads / num_kv_heads, "Attention heads per KV head")?;
        checked_i32(num_q_heads, "Attention query head count")?;
        checked_i32(seq_len.checked_mul(4).ok_or("Attention shared memory size overflow")?, "Attention shared memory size")?;

        let mut output_dev = stream.alloc_zeros::<f32>(q_len).map_err(|e| format!("Allocating attention output failed: {e}"))?;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut launch = stream.launch_builder(&self.function);
        launch.arg(q);
        launch.arg(k_cache);
        launch.arg(v_cache);
        launch.arg(&mut output_dev);
        launch.arg(&seq_i32);
        launch.arg(&kv_i32);
        launch.arg(&head_i32);
        launch.arg(&heads_per_kv);
        launch.arg(&scale);
        let cfg = LaunchConfig { grid_dim: (num_q_heads as u32, 1, 1), block_dim: (THREADS, 1, 1), shared_mem_bytes: (seq_len * 4) as u32 };
        unsafe { launch.launch(cfg) }.map_err(|e| format!("Launching attention CUDA kernel failed: {e}"))?;
        Ok(output_dev)
    }
}

pub struct VectorKernels {
    _module: Arc<CudaModule>,
    add: CudaFunction,
    embed: CudaFunction,
    rope: CudaFunction,
    append_cache: CudaFunction,
}

impl VectorKernels {
    pub fn new(context: &Arc<CudaContext>) -> Result<Self, String> {
        let (module, _) = load_kernel(context, VECTOR_CUDA, "vector_add")?;
        let add = module.load_function("vector_add").map_err(|e| format!("Loading vector_add failed: {e}"))?;
        let embed = module.load_function("embedding_lookup").map_err(|e| format!("Loading embedding_lookup failed: {e}"))?;
        let rope = module.load_function("apply_rope") .map_err(|e| format!("Loading apply_rope failed: {e}"))?;
        let append_cache = module.load_function("append_cache").map_err(|e| format!("Loading append_cache failed: {e}"))?;
        Ok(Self { _module: module, add, embed, rope, append_cache })
    }

    pub fn add(&self, stream: &Arc<CudaStream>, a: &CudaSlice<f32>, b: &CudaSlice<f32>) -> Result<CudaSlice<f32>, String> {
        if a.len() != b.len() || a.is_empty() { return Err("Vector add shape mismatch".into()); }
        let n = checked_i32(a.len(), "Vector add length")?;
        let mut out = stream.alloc_zeros::<f32>(a.len()).map_err(|e| format!("Allocating vector add output failed: {e}"))?;
        let mut launch = stream.launch_builder(&self.add);
        launch.arg(a); launch.arg(b); launch.arg(&mut out); launch.arg(&n);
        let grid = (a.len().div_ceil(THREADS as usize)) as u32;
        unsafe { launch.launch(LaunchConfig { grid_dim: (grid,1,1), block_dim:(THREADS,1,1), shared_mem_bytes:0 }) }.map_err(|e| format!("Launching vector add failed: {e}"))?;
        Ok(out)
    }

    pub fn embedding(&self, stream: &Arc<CudaStream>, table: &CudaSlice<f32>, token: usize, hidden: usize) -> Result<CudaSlice<f32>, String> {
        if hidden == 0 || table.len() % hidden != 0 || token >= table.len()/hidden { return Err("Embedding token or table shape is invalid".into()); }
        let token = checked_i32(token, "Token id")?; let n = checked_i32(hidden, "Hidden size")?;
        let mut out = stream.alloc_zeros::<f32>(hidden).map_err(|e| format!("Allocating embedding output failed: {e}"))?;
        let mut launch = stream.launch_builder(&self.embed);
        launch.arg(table); launch.arg(&mut out); launch.arg(&token); launch.arg(&n);
        unsafe { launch.launch(LaunchConfig { grid_dim: (1,1,1), block_dim:(THREADS,1,1), shared_mem_bytes:0 }) }.map_err(|e| format!("Launching embedding lookup failed: {e}"))?;
        Ok(out)
    }

    pub fn rope(&self, stream: &Arc<CudaStream>, q: &CudaSlice<f32>, k: &CudaSlice<f32>, pos: usize, q_heads: usize, kv_heads: usize, dim: usize, theta: f32) -> Result<(CudaSlice<f32>, CudaSlice<f32>), String> {
        if dim % 2 != 0 || q.len() != q_heads*dim || k.len() != kv_heads*dim { return Err("RoPE tensor shape mismatch".into()); }
        let mut q_out = stream.alloc_zeros::<f32>(q.len()).map_err(|e| format!("Allocating RoPE query failed: {e}"))?;
        let mut k_out = stream.alloc_zeros::<f32>(k.len()).map_err(|e| format!("Allocating RoPE key failed: {e}"))?;
        let (pos, qh, kh, d) = (checked_i32(pos,"RoPE position")?, checked_i32(q_heads,"query heads")?, checked_i32(kv_heads,"KV heads")?, checked_i32(dim,"head dimension")?);
        let mut launch = stream.launch_builder(&self.rope);
        launch.arg(q); launch.arg(k); launch.arg(&mut q_out); launch.arg(&mut k_out); launch.arg(&pos); launch.arg(&qh); launch.arg(&kh); launch.arg(&d); launch.arg(&theta);
        let total = q.len().max(k.len());
        let grid = (total.div_ceil(THREADS as usize)) as u32;
        unsafe { launch.launch(LaunchConfig { grid_dim:(grid,1,1), block_dim:(THREADS,1,1), shared_mem_bytes:0 }) }.map_err(|e| format!("Launching RoPE failed: {e}"))?;
        Ok((q_out,k_out))
    }

    pub fn append_cache(&self, stream: &Arc<CudaStream>, k: &CudaSlice<f32>, v: &CudaSlice<f32>, cache_k: &mut CudaSlice<f32>, cache_v: &mut CudaSlice<f32>, pos: usize, kv_dim: usize) -> Result<(), String> {
        if k.len()!=kv_dim || v.len()!=kv_dim || pos.checked_add(1).and_then(|n|n.checked_mul(kv_dim)).map_or(true, |n| n>cache_k.len() || n>cache_v.len()) { return Err("KV cache capacity or shape is invalid".into()); }
        let pos = checked_i32(pos,"KV cache position")?; let n = checked_i32(kv_dim,"KV dimension")?;
        let mut launch=stream.launch_builder(&self.append_cache);
        launch.arg(k); launch.arg(v); launch.arg(cache_k); launch.arg(cache_v); launch.arg(&pos); launch.arg(&n);
        let grid=(kv_dim.div_ceil(THREADS as usize)) as u32;
        unsafe { launch.launch(LaunchConfig { grid_dim:(grid,1,1), block_dim:(THREADS,1,1), shared_mem_bytes:0 }) }.map_err(|e|format!("Appending KV cache failed: {e}"))?;
        Ok(())
    }
}
