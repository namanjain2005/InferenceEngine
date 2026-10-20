use crate::config::ModelConfig;
use crate::kernels::LinearOp;
use crate::kernels::LinearWeights;
use crate::kernels::ops::{AttentionKernel, RmsNormKernel, SwiGluKernel, VectorKernels};
use crate::kv_cache::KVCacheShape;
use crate::safetensors::SafeTensors;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

pub use crate::kv_cache::{DEFAULT_KV_CACHE_CAPACITY, KVCache, KVCacheConfig};

pub struct LayerWeights {
    input_layernorm: CudaSlice<f32>,
    q_proj_weight: LinearWeights,
    q_proj_bias: CudaSlice<f32>,
    k_proj_weight: LinearWeights,
    k_proj_bias: CudaSlice<f32>,
    v_proj_weight: LinearWeights,
    v_proj_bias: CudaSlice<f32>,
    o_proj_weight: LinearWeights,
    post_attention_layernorm: CudaSlice<f32>,
    gate_proj_weight: LinearWeights,
    up_proj_weight: LinearWeights,
    down_proj_weight: LinearWeights,
}

pub struct Qwen2Model {
    pub config: ModelConfig,
    embed_tokens: CudaSlice<f32>,
    lm_head: Option<LinearWeights>,
    layers: Vec<LayerWeights>,
    norm: CudaSlice<f32>,
    cache_config: KVCacheConfig,
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

fn load_layer_f32(
    st: &mut SafeTensors,
    linear: &dyn LinearOp,
    stream: &Arc<CudaStream>,
    layer: usize,
    name: &str,
) -> Result<CudaSlice<f32>, String> {
    let values = st.read_tensor_f32(&format!("model.layers.{layer}.{name}"))?;
    linear.upload(stream, &values, name)
}

fn load_layer_projection(
    st: &mut SafeTensors,
    linear: &dyn LinearOp,
    stream: &Arc<CudaStream>,
    layer: usize,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<LinearWeights, String> {
    let values = st.read_tensor_f32(&format!("model.layers.{layer}.{name}"))?;
    linear.upload_weight(stream, &values, rows, cols, name)
}

impl Qwen2Model {
    pub fn new_cache_with_capacity(
        &self,
        stream: &Arc<CudaStream>,
        capacity: usize,
    ) -> Result<KVCache, String> {
        self.cache_config
            .with_initial_capacity(capacity)?
            .create_cache(stream, self.cache_shape()?)
    }

    pub fn cache_strategy_name(&self) -> &'static str {
        self.cache_config.strategy_name()
    }

    fn create_cache(&self, stream: &Arc<CudaStream>) -> Result<KVCache, String> {
        self.cache_config.create_cache(stream, self.cache_shape()?)
    }

    fn cache_shape(&self) -> Result<KVCacheShape, String> {
        let kv_dim = self
            .config
            .num_key_value_heads
            .checked_mul(self.config.head_dim())
            .ok_or("KV dimension overflow")?;
        Ok(KVCacheShape {
            num_layers: self.config.num_hidden_layers,
            kv_dim,
        })
    }

    pub fn load(
        st: &mut SafeTensors,
        config: ModelConfig,
        linear: &dyn LinearOp,
        stream: &Arc<CudaStream>,
    ) -> Result<Self, String> {
        Self::load_with_cache_config(st, config, linear, stream, KVCacheConfig::default())
    }

    pub fn load_with_cache_config(
        st: &mut SafeTensors,
        config: ModelConfig,
        linear: &dyn LinearOp,
        stream: &Arc<CudaStream>,
        cache_config: KVCacheConfig,
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
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            let values = st.read_tensor_f32("lm_head.weight")?;
            Some(linear.upload_weight(
                stream,
                &values,
                config.vocab_size,
                config.hidden_size,
                "lm_head.weight",
            )?)
        };
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            print!(
                "\r   Loading and uploading layer weights: {}/{}",
                l + 1,
                config.num_hidden_layers
            );
            use std::io::Write;
            let _ = std::io::stdout().flush();
            layers.push(LayerWeights {
                input_layernorm: load_layer_f32(st, linear, stream, l, "input_layernorm.weight")?,
                q_proj_weight: load_layer_projection(
                    st,
                    linear,
                    stream,
                    l,
                    "self_attn.q_proj.weight",
                    config.num_attention_heads * config.head_dim(),
                    config.hidden_size,
                )?,
                q_proj_bias: load_layer_f32(st, linear, stream, l, "self_attn.q_proj.bias")?,
                k_proj_weight: load_layer_projection(
                    st,
                    linear,
                    stream,
                    l,
                    "self_attn.k_proj.weight",
                    config.num_key_value_heads * config.head_dim(),
                    config.hidden_size,
                )?,
                k_proj_bias: load_layer_f32(st, linear, stream, l, "self_attn.k_proj.bias")?,
                v_proj_weight: load_layer_projection(
                    st,
                    linear,
                    stream,
                    l,
                    "self_attn.v_proj.weight",
                    config.num_key_value_heads * config.head_dim(),
                    config.hidden_size,
                )?,
                v_proj_bias: load_layer_f32(st, linear, stream, l, "self_attn.v_proj.bias")?,
                o_proj_weight: load_layer_projection(
                    st,
                    linear,
                    stream,
                    l,
                    "self_attn.o_proj.weight",
                    config.hidden_size,
                    config.num_attention_heads * config.head_dim(),
                )?,
                post_attention_layernorm: load_layer_f32(
                    st,
                    linear,
                    stream,
                    l,
                    "post_attention_layernorm.weight",
                )?,
                gate_proj_weight: load_layer_projection(
                    st,
                    linear,
                    stream,
                    l,
                    "mlp.gate_proj.weight",
                    config.intermediate_size,
                    config.hidden_size,
                )?,
                up_proj_weight: load_layer_projection(
                    st,
                    linear,
                    stream,
                    l,
                    "mlp.up_proj.weight",
                    config.intermediate_size,
                    config.hidden_size,
                )?,
                down_proj_weight: load_layer_projection(
                    st,
                    linear,
                    stream,
                    l,
                    "mlp.down_proj.weight",
                    config.hidden_size,
                    config.intermediate_size,
                )?,
            });
        }
        println!("\n   All model weights are resident on the GPU.");
        Ok(Self {
            config,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cache_config,
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
            let q = linear.gemv_weight(
                stream,
                &x_norm,
                &layer.q_proj_weight,
                Some(&layer.q_proj_bias),
                hidden,
                q_dim,
            )?;
            let k = linear.gemv_weight(
                stream,
                &x_norm,
                &layer.k_proj_weight,
                Some(&layer.k_proj_bias),
                hidden,
                kv_dim,
            )?;
            let v = linear.gemv_weight(
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
            cache.append(l, pos, &k, &v, vector, stream)?;
            let attn_out = cache.attend_decode(
                l,
                &q,
                pos + 1,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim(),
                attention,
                stream,
            )?;
            let attn_proj =
                linear.gemv_weight(stream, &attn_out, &layer.o_proj_weight, None, q_dim, hidden)?;
            x = vector.add(stream, &x, &attn_proj)?;

            let mlp_norm = rmsnorm.forward(
                stream,
                &x,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
            )?;
            let intermediate = self.config.intermediate_size;
            let gate = linear.gemv_weight(
                stream,
                &mlp_norm,
                &layer.gate_proj_weight,
                None,
                hidden,
                intermediate,
            )?;
            let up = linear.gemv_weight(
                stream,
                &mlp_norm,
                &layer.up_proj_weight,
                None,
                hidden,
                intermediate,
            )?;
            let act = swiglu.forward(stream, &gate, &up)?;
            let mlp_out = linear.gemv_weight(
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

    /// Run one autoregressive decode step for independent requests together.
    /// Projection, normalization, MLP, and logits operations use a GPU batch;
    /// each row keeps and attends to its own KV cache and absolute position.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decode_batch(
        &self,
        token_ids: &[usize],
        positions: &[usize],
        caches: &mut [KVCache],
        linear: &dyn LinearOp,
        rmsnorm: &RmsNormKernel,
        swiglu: &SwiGluKernel,
        attention: &AttentionKernel,
        vector: &VectorKernels,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<f32>, String> {
        let batch = token_ids.len();
        if batch == 0 || positions.len() != batch || caches.len() != batch {
            return Err(
                "Decode batch requires matching non-empty token, position, and cache lists".into(),
            );
        }
        let hidden = self.config.hidden_size;
        let q_dim = self.config.num_attention_heads * self.config.head_dim();
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim();
        for (cache, &pos) in caches.iter_mut().zip(positions) {
            cache.ensure_capacity(
                stream,
                pos.checked_add(1).ok_or("Token position overflow")?,
                pos,
            )?;
        }
        let mut x = vector.embedding_batch(stream, &self.embed_tokens, token_ids, hidden)?;

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let x_norm = rmsnorm.forward_batch(
                stream,
                &x,
                &layer.input_layernorm,
                batch,
                hidden,
                self.config.rms_norm_eps,
            )?;
            let q = linear.gemm_weight(
                stream,
                &x_norm,
                &layer.q_proj_weight,
                Some(&layer.q_proj_bias),
                batch,
                hidden,
                q_dim,
            )?;
            let k = linear.gemm_weight(
                stream,
                &x_norm,
                &layer.k_proj_weight,
                Some(&layer.k_proj_bias),
                batch,
                hidden,
                kv_dim,
            )?;
            let v = linear.gemm_weight(
                stream,
                &x_norm,
                &layer.v_proj_weight,
                Some(&layer.v_proj_bias),
                batch,
                hidden,
                kv_dim,
            )?;
            let (q, k) = vector.rope_batch_positions(
                stream,
                &q,
                &k,
                positions,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim(),
                self.config.rope_theta,
            )?;
            let mut attn_batch = stream
                .alloc_zeros::<f32>(batch * q_dim)
                .map_err(|e| format!("Allocating request-batch attention output failed: {e}"))?;
            for row in 0..batch {
                let q_row = vector.gather_row(stream, &q, row, batch, q_dim)?;
                let k_row = vector.gather_row(stream, &k, row, batch, kv_dim)?;
                let v_row = vector.gather_row(stream, &v, row, batch, kv_dim)?;
                let cache = &mut caches[row];
                cache.append(layer_index, positions[row], &k_row, &v_row, vector, stream)?;
                let attn = cache.attend_decode(
                    layer_index,
                    &q_row,
                    positions[row] + 1,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim(),
                    attention,
                    stream,
                )?;
                stream
                    .memcpy_dtod(
                        &attn,
                        &mut attn_batch.slice_mut(row * q_dim..(row + 1) * q_dim),
                    )
                    .map_err(|e| format!("Copying request attention row failed: {e}"))?;
            }
            let attn_proj = linear.gemm_weight(
                stream,
                &attn_batch,
                &layer.o_proj_weight,
                None,
                batch,
                q_dim,
                hidden,
            )?;
            x = vector.add(stream, &x, &attn_proj)?;
            let mlp_norm = rmsnorm.forward_batch(
                stream,
                &x,
                &layer.post_attention_layernorm,
                batch,
                hidden,
                self.config.rms_norm_eps,
            )?;
            let intermediate = self.config.intermediate_size;
            let gate = linear.gemm_weight(
                stream,
                &mlp_norm,
                &layer.gate_proj_weight,
                None,
                batch,
                hidden,
                intermediate,
            )?;
            let up = linear.gemm_weight(
                stream,
                &mlp_norm,
                &layer.up_proj_weight,
                None,
                batch,
                hidden,
                intermediate,
            )?;
            let act = swiglu.forward(stream, &gate, &up)?;
            let mlp_out = linear.gemm_weight(
                stream,
                &act,
                &layer.down_proj_weight,
                None,
                batch,
                intermediate,
                hidden,
            )?;
            x = vector.add(stream, &x, &mlp_out)?;
        }
        let final_norm = rmsnorm.forward_batch(
            stream,
            &x,
            &self.norm,
            batch,
            hidden,
            self.config.rms_norm_eps,
        )?;
        match &self.lm_head {
            Some(weight) => linear.gemm_weight(
                stream,
                &final_norm,
                weight,
                None,
                batch,
                hidden,
                self.config.vocab_size,
            ),
            None => linear.gemm(
                stream,
                &final_norm,
                &self.embed_tokens,
                None,
                batch,
                hidden,
                self.config.vocab_size,
            ),
        }
    }
}

impl ModelBackend for Qwen2Model {
    type Cache = KVCache;

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn new_cache(&self, stream: &Arc<CudaStream>) -> Result<Self::Cache, String> {
        // Ensure the configured strategy creates state aligned with model layers.
        let cache = self.create_cache(stream)?;
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
        let x = self.forward_core(
            token_id, pos, cache, linear, rmsnorm, swiglu, attention, vector, stream,
        )?;
        let final_norm = rmsnorm.forward(stream, &x, &self.norm, self.config.rms_norm_eps)?;
        let logits = match &self.lm_head {
            Some(weight) => linear.gemv_weight(
                stream,
                &final_norm,
                weight,
                None,
                hidden,
                self.config.vocab_size,
            )?,
            None => linear.gemv(
                stream,
                &final_norm,
                &self.embed_tokens,
                None,
                hidden,
                self.config.vocab_size,
            )?,
        };
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
        self.forward_core(
            token_id, pos, cache, linear, rmsnorm, swiglu, attention, vector, stream,
        )
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
            return self.forward_token(
                tokens[0], start_pos, cache, linear, rmsnorm, swiglu, attention, vector, stream,
            );
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
        let lin = |stream: &Arc<CudaStream>,
                   input: &CudaSlice<f32>,
                   weights: &LinearWeights,
                   bias: Option<&CudaSlice<f32>>,
                   in_f: usize,
                   out_f: usize|
         -> Result<CudaSlice<f32>, String> {
            if use_gemm {
                linear.gemm_weight(stream, input, weights, bias, batch, in_f, out_f)
            } else {
                linear.gemv_batched_weight(stream, input, weights, bias, batch, in_f, out_f)
            }
        };

        let mut x = vector.embedding_batch(stream, &self.embed_tokens, tokens, hidden)?;

        for (l, layer) in self.layers.iter().enumerate() {
            let x_norm = rmsnorm.forward_batch(
                stream,
                &x,
                &layer.input_layernorm,
                batch,
                hidden,
                self.config.rms_norm_eps,
            )?;
            let q = lin(
                stream,
                &x_norm,
                &layer.q_proj_weight,
                Some(&layer.q_proj_bias),
                hidden,
                q_dim,
            )?;
            let k = lin(
                stream,
                &x_norm,
                &layer.k_proj_weight,
                Some(&layer.k_proj_bias),
                hidden,
                kv_dim,
            )?;
            let v = lin(
                stream,
                &x_norm,
                &layer.v_proj_weight,
                Some(&layer.v_proj_bias),
                hidden,
                kv_dim,
            )?;
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
            cache.append_prompt(l, start_pos, batch, &k, &v, vector, stream)?;
            let attn_out = cache.attend_prefill(
                l,
                &q,
                start_pos,
                batch,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim(),
                attention,
                stream,
            )?;
            let attn_proj = lin(stream, &attn_out, &layer.o_proj_weight, None, q_dim, hidden)?;
            x = vector.add(stream, &x, &attn_proj)?;

            let mlp_norm = rmsnorm.forward_batch(
                stream,
                &x,
                &layer.post_attention_layernorm,
                batch,
                hidden,
                self.config.rms_norm_eps,
            )?;
            let gate = lin(
                stream,
                &mlp_norm,
                &layer.gate_proj_weight,
                None,
                hidden,
                intermediate,
            )?;
            let up = lin(
                stream,
                &mlp_norm,
                &layer.up_proj_weight,
                None,
                hidden,
                intermediate,
            )?;
            let act = swiglu.forward(stream, &gate, &up)?;
            let mlp_out = lin(
                stream,
                &act,
                &layer.down_proj_weight,
                None,
                intermediate,
                hidden,
            )?;
            x = vector.add(stream, &x, &mlp_out)?;
        }
        // Logits for the last prompt token only: avoids a [batch, vocab]
        // temporary (batch * 151936 floats) and matches forward_token output.
        let final_norm = rmsnorm.forward_batch(
            stream,
            &x,
            &self.norm,
            batch,
            hidden,
            self.config.rms_norm_eps,
        )?;
        let last = vector.gather_row(stream, &final_norm, batch - 1, batch, hidden)?;
        let logits = match &self.lm_head {
            Some(weight) => {
                linear.gemv_weight(stream, &last, weight, None, hidden, self.config.vocab_size)?
            }
            None => linear.gemv(
                stream,
                &last,
                &self.embed_tokens,
                None,
                hidden,
                self.config.vocab_size,
            )?,
        };
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
    if !config.tie_word_embeddings {
        check("lm_head.weight".into(), &[config.vocab_size, hidden])?;
    }
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
