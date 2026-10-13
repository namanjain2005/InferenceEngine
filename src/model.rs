use crate::config::ModelConfig;
use crate::kernels::linear::LinearKernel;
use crate::kernels::ops::{AttentionKernel, RmsNormKernel, SwiGluKernel, VectorKernels};
use crate::safetensors::SafeTensors;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

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
    fn new_cache(&self, stream: &Arc<CudaStream>, capacity: usize) -> Result<Self::Cache, String>;
    fn forward_token(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut Self::Cache,
        linear: &LinearKernel,
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
        linear: &LinearKernel,
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
}

impl KVCache {
    pub fn new(
        stream: &Arc<CudaStream>,
        num_layers: usize,
        capacity: usize,
        kv_dim: usize,
    ) -> Result<Self, String> {
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
        Ok(Self { layers })
    }
}

impl Qwen2Model {
    pub fn load(
        st: &mut SafeTensors,
        config: ModelConfig,
        linear: &LinearKernel,
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
}

impl ModelBackend for Qwen2Model {
    type Cache = KVCache;

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_cache(&self, stream: &Arc<CudaStream>, capacity: usize) -> Result<Self::Cache, String> {
        let kv_dim = self
            .config
            .num_key_value_heads
            .checked_mul(self.config.head_dim())
            .ok_or("KV dimension overflow")?;
        KVCache::new(stream, self.config.num_hidden_layers, capacity, kv_dim)
    }

    fn forward_token(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut KVCache,
        linear: &LinearKernel,
        rmsnorm: &RmsNormKernel,
        swiglu: &SwiGluKernel,
        attention: &AttentionKernel,
        vector: &VectorKernels,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<f32>, String> {
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
            let q = linear.forward(
                stream,
                &x_norm,
                &layer.q_proj_weight,
                Some(&layer.q_proj_bias),
                1,
                hidden,
                q_dim,
            )?;
            let k = linear.forward(
                stream,
                &x_norm,
                &layer.k_proj_weight,
                Some(&layer.k_proj_bias),
                1,
                hidden,
                kv_dim,
            )?;
            let v = linear.forward(
                stream,
                &x_norm,
                &layer.v_proj_weight,
                Some(&layer.v_proj_bias),
                1,
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
            let attn_proj = linear.forward(
                stream,
                &attn_out,
                &layer.o_proj_weight,
                None,
                1,
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
            let gate = linear.forward(
                stream,
                &mlp_norm,
                &layer.gate_proj_weight,
                None,
                1,
                hidden,
                intermediate,
            )?;
            let up = linear.forward(
                stream,
                &mlp_norm,
                &layer.up_proj_weight,
                None,
                1,
                hidden,
                intermediate,
            )?;
            let act = swiglu.forward(stream, &gate, &up)?;
            let mlp_out = linear.forward(
                stream,
                &act,
                &layer.down_proj_weight,
                None,
                1,
                intermediate,
                hidden,
            )?;
            x = vector.add(stream, &x, &mlp_out)?;
        }
        let final_norm = rmsnorm.forward(stream, &x, &self.norm, self.config.rms_norm_eps)?;
        let logits = linear.forward(
            stream,
            &final_norm,
            &self.embed_tokens,
            None,
            1,
            hidden,
            self.config.vocab_size,
        )?;
        Ok(logits)
    }

    fn forward_prefill(
        &self,
        tokens: &[usize],
        start_pos: usize,
        cache: &mut KVCache,
        linear: &LinearKernel,
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
                linear.forward_gemm(stream, input, weights, bias, batch, in_f, out_f)
            } else {
                linear.forward(stream, input, weights, bias, batch, in_f, out_f)
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
        let logits = linear.forward(stream, &last, &self.embed_tokens, None, 1, hidden, self.config.vocab_size)?;
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
