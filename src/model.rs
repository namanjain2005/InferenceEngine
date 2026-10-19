use crate::config::ModelConfig;
use crate::kernels::LinearOp;
use crate::kernels::ops::{AttentionKernel, RmsNormKernel, SwiGluKernel, VectorKernels};
use crate::safetensors::SafeTensors;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

/// Initial number of token positions a [`KVCache`] allocates. The cache grows
/// geometrically on demand, so this only reserves memory for the common short
/// prompt case and never imposes a hard sequence-length limit.
pub const DEFAULT_KV_CACHE_CAPACITY: usize = 2048;

pub struct LayerWeights {
    input_layernorm: CudaSlice<f32>,
    q_proj_weight: CudaSlice<f32>,
    q_proj_bias: CudaSlice<f32>,
    k_proj_weight: CudaSlice<f32>,
    k_proj_bias: CudaSlice<f32>,
    v_proj_weight: CudaSlice<f32>,
    v_proj_bias: CudaSlice<f32>,
    o_proj_weight: CudaSlice<f32>,
    post_attention_layernorm: CudaSlice<f32>,
    gate_proj_weight: CudaSlice<f32>,
    up_proj_weight: CudaSlice<f32>,
    down_proj_weight: CudaSlice<f32>,
}

pub struct Qwen2Model {
    pub config: ModelConfig,
    embed_tokens: CudaSlice<f32>,
    layers: Vec<LayerWeights>,
    norm: CudaSlice<f32>,
}

/// Backend contract for causal decoder models. Each backend owns its cache
/// layout and maps its checkpoint tensors to the shared inference kernels.
pub trait ModelBackend {
    type Cache;
    fn config(&self) -> &ModelConfig;
    fn new_cache(&self, stream: &Arc<CudaStream>) -> Result<Self::Cache, String>;
    fn forward_token(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut Self::Cache,
        linear: &dyn LinearOp,
        rmsnorm: &RmsNormKernel,
        swiglu: &SwiGluKernel,
        attention: &AttentionKernel,
        vector: &VectorKernels,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<f32>, String>;
    /// Token-at-a-time forward without the final norm/logits projection.
    /// Returns the last residual stream vector. Used for prefill positions
    /// whose logits are discarded, so the loop baseline does the same work
    /// as the batched paths (which project only the last row).
    fn forward_token_hidden(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut Self::Cache,
        linear: &dyn LinearOp,
        rmsnorm: &RmsNormKernel,
        swiglu: &SwiGluKernel,
        attention: &AttentionKernel,
        vector: &VectorKernels,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<f32>, String>;
    /// Batched prefill over `tokens` at cache positions
    /// `[start_pos, start_pos + tokens.len())`. Fills the KV cache and returns
    /// logits for the last prompt token only. `use_gemm` selects the blocked
    /// GEMM linear path (`true`) or the batched GEMV baseline (`false`);
    /// both read the same resident weight slices.
    fn forward_prefill(
        &self,
        tokens: &[usize],
        start_pos: usize,
        cache: &mut Self::Cache,
        linear: &dyn LinearOp,
        rmsnorm: &RmsNormKernel,
        swiglu: &SwiGluKernel,
        attention: &AttentionKernel,
        vector: &VectorKernels,
        stream: &Arc<CudaStream>,
        use_gemm: bool,
    ) -> Result<CudaSlice<f32>, String>;
}

pub struct LayerKVCache {
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
}
pub struct KVCache {
    layers: Vec<LayerKVCache>,
    capacity: usize,
    kv_dim: usize,
}

impl KVCache {
    /// Allocate a cache using [`DEFAULT_KV_CACHE_CAPACITY`] positions.
    pub fn new(
        stream: &Arc<CudaStream>,
        num_layers: usize,
        kv_dim: usize,
    ) -> Result<Self, String> {
        Self::with_capacity(stream, num_layers, DEFAULT_KV_CACHE_CAPACITY, kv_dim)
    }

    /// Allocate a cache with an explicit initial number of token positions.
    /// Each layer stores `capacity * kv_dim` key and value floats. The cache
    /// still grows automatically if generation runs past this capacity.
    pub fn with_capacity(
        stream: &Arc<CudaStream>,
        num_layers: usize,
        capacity: usize,
        kv_dim: usize,
    ) -> Result<Self, String> {
        if num_layers == 0 || capacity == 0 || kv_dim == 0 {
            return Err("KV cache layers, capacity, and dimension must be non-zero".into());
        }
        let len = capacity
            .checked_mul(kv_dim)
            .ok_or("KV cache size overflow")?;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let k = stream
                .alloc_zeros::<f32>(len)
                .map_err(|e| format!("Allocating GPU key cache failed: {e}"))?;
            let v = stream
                .alloc_zeros::<f32>(len)
                .map_err(|e| format!("Allocating GPU value cache failed: {e}"))?;
            layers.push(LayerKVCache { k, v });
        }
        Ok(Self {
            layers,
            capacity,
            kv_dim,
        })
    }

    /// Ensure at least `needed` token positions fit, growing the cache if
    /// necessary. `preserve` is the number of rows already written and must
    /// keep their contents. Growth is geometric (doubling) so decoding past
    /// the initial capacity reallocates O(log n) times; only the live rows are
    /// copied, and each old buffer is freed as soon as its replacement is in
    /// place.
    fn ensure_capacity(
        &mut self,
        stream: &Arc<CudaStream>,
        needed: usize,
        preserve: usize,
    ) -> Result<(), String> {
        if needed <= self.capacity {
            return Ok(());
        }
        let old_capacity = self.capacity();
        debug_assert!(
            preserve <= old_capacity,
            "cannot preserve {preserve} rows in a {old_capacity}-row cache"
        );
        let doubled = old_capacity
            .checked_mul(2)
            .ok_or("KV cache capacity overflow")?;
        let new_capacity = needed.max(doubled);
        let new_len = new_capacity
            .checked_mul(self.kv_dim)
            .ok_or("KV cache size overflow")?;
        let copy_len = preserve
            .min(old_capacity)
            .checked_mul(self.kv_dim)
            .ok_or("KV cache copy size overflow")?;

        for layer in &mut self.layers {
            let mut k = stream
                .alloc_zeros::<f32>(new_len)
                .map_err(|e| format!("Growing GPU key cache failed: {e}"))?;
            let mut v = stream
                .alloc_zeros::<f32>(new_len)
                .map_err(|e| format!("Growing GPU value cache failed: {e}"))?;
            if copy_len > 0 {
                stream
                    .memcpy_dtod(&layer.k.slice(..copy_len), &mut k.slice_mut(..copy_len))
                    .map_err(|e| format!("Copying existing key cache failed: {e}"))?;
                stream
                    .memcpy_dtod(&layer.v.slice(..copy_len), &mut v.slice_mut(..copy_len))
                    .map_err(|e| format!("Copying existing value cache failed: {e}"))?;
            }
            // Old buffers are dropped (freed) here; their contents are no
            // longer referenced by subsequent kernels.
            layer.k = k;
            layer.v = v;
        }
        self.capacity = new_capacity;
        Ok(())
    }

    /// Number of token positions this cache can currently hold. The value can
    /// increase during generation as the cache grows.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of transformer layers backed by this cache.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl Qwen2Model {
    pub fn load(
        st: &mut SafeTensors,
        config: ModelConfig,
        linear: &dyn LinearOp,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, String> {
        println!("   Uploading embedding and final norm weights to GPU...");
        let embed_tokens = linear.upload(
            stream,
            &st.read_tensor_f32("model.embed_tokens.weight")?,
            "embedding weights",
        )?;
        let norm = linear.upload(
            stream,
            &st.read_tensor_f32("model.norm.weight")?,
            "final norm weights",
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            print!(
                "\r   Loading and uploading layer weights: {}/{}",
                l + 1,
                config.num_hidden_layers
            );
            use std::io::Write;
            let _ = std::io::stdout().flush();
            let mut load = |name: &str| -> Result<CudaSlice<f32>, String> {
                let values = st.read_tensor_f32(&format!("model.layers.{l}.{name}"))?;
                linear.upload(stream, &values, name)
            };
            layers.push(LayerWeights {
                input_layernorm: load("input_layernorm.weight")?,
                q_proj_weight: load("self_attn.q_proj.weight")?,
                q_proj_bias: load("self_attn.q_proj.bias")?,
                k_proj_weight: load("self_attn.k_proj.weight")?,
                k_proj_bias: load("self_attn.k_proj.bias")?,
                v_proj_weight: load("self_attn.v_proj.weight")?,
                v_proj_bias: load("self_attn.v_proj.bias")?,
                o_proj_weight: load("self_attn.o_proj.weight")?,
                post_attention_layernorm: load("post_attention_layernorm.weight")?,
                gate_proj_weight: load("mlp.gate_proj.weight")?,
                up_proj_weight: load("mlp.up_proj.weight")?,
                down_proj_weight: load("mlp.down_proj.weight")?,
            });
        }
        println!("\n   All model weights are resident on the GPU.");
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Shared token-at-a-time body through the last residual stream vector.
    /// `forward_token` adds the final norm + vocab projection;
    /// `forward_token_hidden` returns the residual directly.

    fn forward_core(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut KVCache,
        linear: &dyn LinearOp,
        rmsnorm: &RmsNormKernel,
        swiglu: &SwiGluKernel,
        attention: &AttentionKernel,
        vector: &VectorKernels,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<f32>, String> {
        let next_pos = pos.checked_add(1).ok_or("Token position overflow")?;
        cache.ensure_capacity(stream, next_pos, pos)?;
        let hidden = self.config.hidden_size;
        let q_dim = self
            .config
            .num_attention_heads
            .checked_mul(self.config.head_dim())
            .ok_or("Query dimension overflow")?;
        let kv_dim = self
            .config
            .num_key_value_heads
            .checked_mul(self.config.head_dim())
            .ok_or("KV dimension overflow")?;
        let mut x = vector.embedding(stream, &self.embed_tokens, token_id, hidden)?;

        for (l, layer) in self.layers.iter().enumerate() {
            let x_norm =
                rmsnorm.forward(stream, &x, &layer.input_layernorm, self.config.rms_norm_eps)?;
            let q = linear.gemv(
                stream,
                &x_norm,
                &layer.q_proj_weight,
                Some(&layer.q_proj_bias),
                hidden,
                q_dim,
            )?;
            let k = linear.gemv(
                stream,
                &x_norm,
                &layer.k_proj_weight,
                Some(&layer.k_proj_bias),
                hidden,
                kv_dim,
            )?;
            let v = linear.gemv(
                stream,
                &x_norm,
                &layer.v_proj_weight,
                Some(&layer.v_proj_bias),
                hidden,
                kv_dim,
            )?;
            let (q, k) = vector.rope(
                stream,
                &q,
                &k,
                pos,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim(),
                self.config.rope_theta,
            )?;
            let layer_cache = &mut cache.layers[l];
            vector.append_cache(
                stream,
                &k,
                &v,
                &mut layer_cache.k,
                &mut layer_cache.v,
                pos,
                kv_dim,
            )?;
            let attn_out = attention.forward(
                stream,
                &q,
                &layer_cache.k,
                &layer_cache.v,
                pos + 1,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim(),
            )?;
            let attn_proj = linear.gemv(
                stream,
                &attn_out,
                &layer.o_proj_weight,
                None,
                q_dim,
                hidden,
            )?;
            x = vector.add(stream, &x, &attn_proj)?;

            let mlp_norm = rmsnorm.forward(
                stream,
                &x,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;
            let intermediate = self.config.intermediate_size;
            let gate = linear.gemv(
                stream,
                &mlp_norm,
                &layer.gate_proj_weight,
                None,
                hidden,
                intermediate,
            )?;
            let up = linear.gemv(
                stream,
                &mlp_norm,
                &layer.up_proj_weight,
                None,
                hidden,
                intermediate,
            )?;
            let act = swiglu.forward(stream, &gate, &up)?;
            let mlp_out = linear.gemv(
                stream,
                &act,
                &layer.down_proj_weight,
                None,
                intermediate,
                hidden,
            )?;
            x = vector.add(stream, &x, &mlp_out)?;
        }
        Ok(x)
    }

}

impl ModelBackend for Qwen2Model {
    type Cache = KVCache;

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_cache(&self, stream: &Arc<CudaStream>) -> Result<Self::Cache, String> {
        let kv_dim = self
            .config
            .num_key_value_heads
            .checked_mul(self.config.head_dim())
            .ok_or("KV dimension overflow")?;
        // `KVCache::new` applies DEFAULT_KV_CACHE_CAPACITY; keep the cache
        // layer count in lockstep with the resident model layers.
        let cache = KVCache::new(stream, self.config.num_hidden_layers, kv_dim)?;
        if cache.num_layers() != self.layers.len() {
            return Err(format!(
                "KV cache has {} layers but the model loaded {}",
                cache.num_layers(),
                self.layers.len()
            ));
        }
        Ok(cache)
    }

    fn forward_token(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut KVCache,
        linear: &dyn LinearOp,
        rmsnorm: &RmsNormKernel,
        swiglu: &SwiGluKernel,
        attention: &AttentionKernel,
        vector: &VectorKernels,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<f32>, String> {
        let hidden = self.config.hidden_size;
        let x = self.forward_core(token_id, pos, cache, linear, rmsnorm, swiglu, attention, vector, stream)?;
        let final_norm = rmsnorm.forward(stream, &x, &self.norm, self.config.rms_norm_eps)?;
        let logits = linear.gemv(
            stream,
            &final_norm,
            &self.embed_tokens,
            None,
            hidden,
            self.config.vocab_size,
        )?;
        Ok(logits)
    }

    fn forward_token_hidden(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut KVCache,
        linear: &dyn LinearOp,
        rmsnorm: &RmsNormKernel,
        swiglu: &SwiGluKernel,
        attention: &AttentionKernel,
        vector: &VectorKernels,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<f32>, String> {
        self.forward_core(token_id, pos, cache, linear, rmsnorm, swiglu, attention, vector, stream)
    }

    fn forward_prefill(
        &self,
        tokens: &[usize],
        start_pos: usize,
        cache: &mut KVCache,
        linear: &dyn LinearOp,
        rmsnorm: &RmsNormKernel,
        swiglu: &SwiGluKernel,
        attention: &AttentionKernel,
        vector: &VectorKernels,
        stream: &Arc<CudaStream>,
        use_gemm: bool,
    ) -> Result<CudaSlice<f32>, String> {
        let batch = tokens.len();
        if batch == 0 {
            return Err("Prefill token batch must not be empty".into());
        }
        let end_pos = start_pos
            .checked_add(batch)
            .ok_or("Prefill sequence length overflow")?;
        // Grow the cache before any kernel launch so appends and attention see
        // one allocation for the whole prefill batch.
        cache.ensure_capacity(stream, end_pos, start_pos)?;
        // Single trailing token: identical to the decode path, no batching win.
        if batch == 1 {
            return self.forward_token(tokens[0], start_pos, cache, linear, rmsnorm, swiglu, attention, vector, stream);
        }
        let hidden = self.config.hidden_size;
        let q_dim = self
            .config
            .num_attention_heads
            .checked_mul(self.config.head_dim())
            .ok_or("Query dimension overflow")?;
        let kv_dim = self
            .config
            .num_key_value_heads
            .checked_mul(self.config.head_dim())
            .ok_or("KV dimension overflow")?;
        let intermediate = self.config.intermediate_size;

        // Batched linear helper: same weights, GEMM or batched-GEMV baseline.
        let lin = |stream: &Arc<CudaStream>, input: &CudaSlice<f32>, weights: &CudaSlice<f32>, bias: Option<&CudaSlice<f32>>, in_f: usize, out_f: usize| -> Result<CudaSlice<f32>, String> {
            if use_gemm {
                linear.gemm(stream, input, weights, bias, batch, in_f, out_f)
            } else {
                linear.gemv_batched(stream, input, weights, bias, batch, in_f, out_f)
            }
        };

        let mut x = vector.embedding_batch(stream, &self.embed_tokens, tokens, hidden)?;

        for (l, layer) in self.layers.iter().enumerate() {
            let x_norm = rmsnorm.forward_batch(stream, &x, &layer.input_layernorm, batch, hidden, self.config.rms_norm_eps)?;
            let q = lin(stream, &x_norm, &layer.q_proj_weight, Some(&layer.q_proj_bias), hidden, q_dim)?;
            let k = lin(stream, &x_norm, &layer.k_proj_weight, Some(&layer.k_proj_bias), hidden, kv_dim)?;
            let v = lin(stream, &x_norm, &layer.v_proj_weight, Some(&layer.v_proj_bias), hidden, kv_dim)?;
            let (q, k) = vector.rope_batch(
                stream,
                &q,
                &k,
                start_pos,
                batch,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim(),
                self.config.rope_theta,
            )?;
            let layer_cache = &mut cache.layers[l];
            vector.append_cache_batch(stream, &k, &v, &mut layer_cache.k, &mut layer_cache.v, start_pos, batch, kv_dim)?;
            let attn_out = attention.forward_prefill(
                stream,
                &q,
                &layer_cache.k,
                &layer_cache.v,
                start_pos,
                batch,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim(),
            )?;
            let attn_proj = lin(stream, &attn_out, &layer.o_proj_weight, None, q_dim, hidden)?;
            x = vector.add(stream, &x, &attn_proj)?;

            let mlp_norm = rmsnorm.forward_batch(stream, &x, &layer.post_attention_layernorm, batch, hidden, self.config.rms_norm_eps)?;
            let gate = lin(stream, &mlp_norm, &layer.gate_proj_weight, None, hidden, intermediate)?;
            let up = lin(stream, &mlp_norm, &layer.up_proj_weight, None, hidden, intermediate)?;
            let act = swiglu.forward(stream, &gate, &up)?;
            let mlp_out = lin(stream, &act, &layer.down_proj_weight, None, intermediate, hidden)?;
            x = vector.add(stream, &x, &mlp_out)?;
        }
        // Logits for the last prompt token only: avoids a [batch, vocab]
        // temporary (batch * 151936 floats) and matches forward_token output.
        let final_norm = rmsnorm.forward_batch(stream, &x, &self.norm, batch, hidden, self.config.rms_norm_eps)?;
        let last = vector.gather_row(stream, &final_norm, batch - 1, batch, hidden)?;
        let logits = linear.gemv(stream, &last, &self.embed_tokens, None, hidden, self.config.vocab_size)?;
        Ok(logits)
    }
}

/// Validate checkpoint tensor names, dtypes, and dimensions without a CUDA device.
pub fn validate_qwen2_checkpoint(
    config: &ModelConfig,
    tensors: &SafeTensors,
) -> Result<(usize, usize), String> {
    config.validate()?;
    let hidden = config.hidden_size;
    let q_dim = config.num_attention_heads * config.head_dim();
    let kv_dim = config.num_key_value_heads * config.head_dim();
    let mut parameter_count = 0usize;
    let mut bytes = 0usize;
    let mut check = |name: String, expected: &[usize]| -> Result<(), String> {
        let meta = tensors
            .get_meta(&name)
            .ok_or_else(|| format!("Required tensor '{name}' is missing"))?;
        if meta.shape != expected {
            return Err(format!(
                "Tensor '{name}' has shape {:?}; expected {expected:?}",
                meta.shape
            ));
        }
        let item_bytes = match meta.dtype.as_str() {
            "BF16" | "F16" => 2,
            "F32" => 4,
            other => return Err(format!("Tensor '{name}' has unsupported dtype {other}")),
        };
        let elements = expected
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| format!("Tensor '{name}' element count overflow"))?;
        parameter_count = parameter_count
            .checked_add(elements)
            .ok_or("Parameter count overflow")?;
        bytes = bytes
            .checked_add(
                elements
                    .checked_mul(item_bytes)
                    .ok_or("Checkpoint byte count overflow")?,
            )
            .ok_or("Checkpoint byte count overflow")?;
        Ok(())
    };

    check(
        "model.embed_tokens.weight".into(),
        &[config.vocab_size, hidden],
    )?;
    check("model.norm.weight".into(), &[hidden])?;
    for layer in 0..config.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        check(format!("{p}.input_layernorm.weight"), &[hidden])?;
        check(format!("{p}.self_attn.q_proj.weight"), &[q_dim, hidden])?;
        check(format!("{p}.self_attn.q_proj.bias"), &[q_dim])?;
        check(format!("{p}.self_attn.k_proj.weight"), &[kv_dim, hidden])?;
        check(format!("{p}.self_attn.k_proj.bias"), &[kv_dim])?;
        check(format!("{p}.self_attn.v_proj.weight"), &[kv_dim, hidden])?;
        check(format!("{p}.self_attn.v_proj.bias"), &[kv_dim])?;
        check(format!("{p}.self_attn.o_proj.weight"), &[hidden, q_dim])?;
        check(format!("{p}.post_attention_layernorm.weight"), &[hidden])?;
        check(
            format!("{p}.mlp.gate_proj.weight"),
            &[config.intermediate_size, hidden],
        )?;
        check(
            format!("{p}.mlp.up_proj.weight"),
            &[config.intermediate_size, hidden],
        )?;
        check(
            format!("{p}.mlp.down_proj.weight"),
            &[hidden, config.intermediate_size],
        )?;
    }
    Ok((parameter_count, bytes))
}
