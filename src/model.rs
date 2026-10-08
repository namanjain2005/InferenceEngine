use crate::config::QwenConfig;
use crate::kernels::linear::LinearKernel;
use crate::kernels::ops::{AttentionKernel, RmsNormKernel, SwiGluKernel, VectorKernels};
use crate::safetensors::SafeTensors;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

pub struct LayerWeights {
    input_layernorm: CudaSlice<f32>, q_proj_weight: CudaSlice<f32>, q_proj_bias: CudaSlice<f32>,
    k_proj_weight: CudaSlice<f32>, k_proj_bias: CudaSlice<f32>, v_proj_weight: CudaSlice<f32>,
    v_proj_bias: CudaSlice<f32>, o_proj_weight: CudaSlice<f32>, post_attention_layernorm: CudaSlice<f32>,
    gate_proj_weight: CudaSlice<f32>, up_proj_weight: CudaSlice<f32>, down_proj_weight: CudaSlice<f32>,
}

pub struct ModelWeights { pub config: QwenConfig, embed_tokens: CudaSlice<f32>, layers: Vec<LayerWeights>, norm: CudaSlice<f32> }

pub struct LayerKVCache { k: CudaSlice<f32>, v: CudaSlice<f32> }
pub struct KVCache { layers: Vec<LayerKVCache> }

impl KVCache {
    pub fn new(stream: &Arc<CudaStream>, num_layers: usize, capacity: usize, kv_dim: usize) -> Result<Self, String> {
        let len = capacity.checked_mul(kv_dim).ok_or("KV cache size overflow")?;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let k = stream.alloc_zeros::<f32>(len).map_err(|e| format!("Allocating GPU key cache failed: {e}"))?;
            let v = stream.alloc_zeros::<f32>(len).map_err(|e| format!("Allocating GPU value cache failed: {e}"))?;
            layers.push(LayerKVCache { k, v });
        }
        Ok(Self { layers })
    }
}

impl ModelWeights {
    pub fn load(st: &mut SafeTensors, config: QwenConfig, linear: &LinearKernel, stream: &Arc<CudaStream>) -> Result<Self, String> {
        println!("   Uploading embedding and final norm weights to GPU...");
        let embed_tokens = linear.upload(stream, &st.read_tensor_f32("model.embed_tokens.weight")?, "embedding weights")?;
        let norm = linear.upload(stream, &st.read_tensor_f32("model.norm.weight")?, "final norm weights")?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            print!("\r   Loading and uploading layer weights: {}/{}", l + 1, config.num_hidden_layers);
            use std::io::Write;
            let _ = std::io::stdout().flush();
            let mut load = |name: &str| -> Result<CudaSlice<f32>, String> {
                let values = st.read_tensor_f32(&format!("model.layers.{l}.{name}"))?;
                linear.upload(stream, &values, name)
            };
            layers.push(LayerWeights {
                input_layernorm: load("input_layernorm.weight")?,
                q_proj_weight: load("self_attn.q_proj.weight")?, q_proj_bias: load("self_attn.q_proj.bias")?,
                k_proj_weight: load("self_attn.k_proj.weight")?, k_proj_bias: load("self_attn.k_proj.bias")?,
                v_proj_weight: load("self_attn.v_proj.weight")?, v_proj_bias: load("self_attn.v_proj.bias")?,
                o_proj_weight: load("self_attn.o_proj.weight")?,
                post_attention_layernorm: load("post_attention_layernorm.weight")?,
                gate_proj_weight: load("mlp.gate_proj.weight")?, up_proj_weight: load("mlp.up_proj.weight")?,
                down_proj_weight: load("mlp.down_proj.weight")?,
            });
        }
        println!("\n   All model weights are resident on the GPU.");
        Ok(Self { config, embed_tokens, layers, norm })
    }

    pub fn forward_token(
        &self, token_id: usize, pos: usize, cache: &mut KVCache, linear: &LinearKernel,
        rmsnorm: &RmsNormKernel, swiglu: &SwiGluKernel, attention: &AttentionKernel,
        vector: &VectorKernels, stream: &Arc<CudaStream>,
    ) -> Result<Vec<f32>, String> {
        let hidden = self.config.hidden_size;
        let q_dim = self.config.num_attention_heads.checked_mul(self.config.head_dim()).ok_or("Query dimension overflow")?;
        let kv_dim = self.config.num_key_value_heads.checked_mul(self.config.head_dim()).ok_or("KV dimension overflow")?;
        let mut x = vector.embedding(stream, &self.embed_tokens, token_id, hidden)?;

        for (l, layer) in self.layers.iter().enumerate() {
            let x_norm = rmsnorm.forward(stream, &x, &layer.input_layernorm, self.config.rms_norm_eps)?;
            let q = linear.forward(stream, &x_norm, &layer.q_proj_weight, Some(&layer.q_proj_bias), 1, hidden, q_dim)?;
            let k = linear.forward(stream, &x_norm, &layer.k_proj_weight, Some(&layer.k_proj_bias), 1, hidden, kv_dim)?;
            let v = linear.forward(stream, &x_norm, &layer.v_proj_weight, Some(&layer.v_proj_bias), 1, hidden, kv_dim)?;
            let (q, k) = vector.rope(stream, &q, &k, pos, self.config.num_attention_heads, self.config.num_key_value_heads, self.config.head_dim(), self.config.rope_theta)?;
            let layer_cache = &mut cache.layers[l];
            vector.append_cache(stream, &k, &v, &mut layer_cache.k, &mut layer_cache.v, pos, kv_dim)?;
            let attn_out = attention.forward(stream, &q, &layer_cache.k, &layer_cache.v, pos + 1,
                self.config.num_attention_heads, self.config.num_key_value_heads, self.config.head_dim())?;
            let attn_proj = linear.forward(stream, &attn_out, &layer.o_proj_weight, None, 1, q_dim, hidden)?;
            x = vector.add(stream, &x, &attn_proj)?;

            let mlp_norm = rmsnorm.forward(stream, &x, &layer.post_attention_layernorm, self.config.rms_norm_eps)?;
            let intermediate = self.config.intermediate_size;
            let gate = linear.forward(stream, &mlp_norm, &layer.gate_proj_weight, None, 1, hidden, intermediate)?;
            let up = linear.forward(stream, &mlp_norm, &layer.up_proj_weight, None, 1, hidden, intermediate)?;
            let act = swiglu.forward(stream, &gate, &up)?;
            let mlp_out = linear.forward(stream, &act, &layer.down_proj_weight, None, 1, intermediate, hidden)?;
            x = vector.add(stream, &x, &mlp_out)?;
        }
        let final_norm = rmsnorm.forward(stream, &x, &self.norm, self.config.rms_norm_eps)?;
        let logits = linear.forward(stream, &final_norm, &self.embed_tokens, None, 1, hidden, self.config.vocab_size)?;
        stream.clone_dtoh(&logits).map_err(|e| format!("Copying final logits to host failed: {e}"))
    }
}
