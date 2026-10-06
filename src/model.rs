use crate::config::QwenConfig;
use crate::safetensors::SafeTensors;
use crate::kernels::linear::LinearKernel;
use cudarc::driver::CudaStream;
use std::sync::Arc;

pub struct LayerWeights {
    pub input_layernorm: Vec<f32>,
    pub q_proj_weight: Vec<f32>,
    pub q_proj_bias: Vec<f32>,
    pub k_proj_weight: Vec<f32>,
    pub k_proj_bias: Vec<f32>,
    pub v_proj_weight: Vec<f32>,
    pub v_proj_bias: Vec<f32>,
    pub o_proj_weight: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub gate_proj_weight: Vec<f32>,
    pub up_proj_weight: Vec<f32>,
    pub down_proj_weight: Vec<f32>,
}

pub struct ModelWeights {
    pub config: QwenConfig,
    pub embed_tokens: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub norm: Vec<f32>,
}

pub struct LayerKVCache {
    pub k: Vec<f32>, // shape: [seq_len, num_kv_heads * head_dim]
    pub v: Vec<f32>, // shape: [seq_len, num_kv_heads * head_dim]
}

pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(LayerKVCache {
                k: Vec::new(),
                v: Vec::new(),
            });
        }
        Self { layers }
    }
}

impl ModelWeights {
    pub fn load(st: &mut SafeTensors, config: QwenConfig) -> Result<Self, String> {
        println!("   Loading embedding weights [{} x {}]...", config.vocab_size, config.hidden_size);
        let embed_tokens = st.read_tensor_f32("model.embed_tokens.weight")?;

        println!("   Loading final layernorm weights [{}]...", config.hidden_size);
        let norm = st.read_tensor_f32("model.norm.weight")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            print!("\r   Loading layer weights: {}/{}", l + 1, config.num_hidden_layers);
            use std::io::Write;
            let _ = std::io::stdout().flush();

            let input_layernorm = st.read_tensor_f32(&format!("model.layers.{}.input_layernorm.weight", l))?;
            let q_proj_weight = st.read_tensor_f32(&format!("model.layers.{}.self_attn.q_proj.weight", l))?;
            let q_proj_bias = st.read_tensor_f32(&format!("model.layers.{}.self_attn.q_proj.bias", l))?;
            let k_proj_weight = st.read_tensor_f32(&format!("model.layers.{}.self_attn.k_proj.weight", l))?;
            let k_proj_bias = st.read_tensor_f32(&format!("model.layers.{}.self_attn.k_proj.bias", l))?;
            let v_proj_weight = st.read_tensor_f32(&format!("model.layers.{}.self_attn.v_proj.weight", l))?;
            let v_proj_bias = st.read_tensor_f32(&format!("model.layers.{}.self_attn.v_proj.bias", l))?;
            let o_proj_weight = st.read_tensor_f32(&format!("model.layers.{}.self_attn.o_proj.weight", l))?;

            let post_attention_layernorm = st.read_tensor_f32(&format!("model.layers.{}.post_attention_layernorm.weight", l))?;
            let gate_proj_weight = st.read_tensor_f32(&format!("model.layers.{}.mlp.gate_proj.weight", l))?;
            let up_proj_weight = st.read_tensor_f32(&format!("model.layers.{}.mlp.up_proj.weight", l))?;
            let down_proj_weight = st.read_tensor_f32(&format!("model.layers.{}.mlp.down_proj.weight", l))?;

            layers.push(LayerWeights {
                input_layernorm,
                q_proj_weight,
                q_proj_bias,
                k_proj_weight,
                k_proj_bias,
                v_proj_weight,
                v_proj_bias,
                o_proj_weight,
                post_attention_layernorm,
                gate_proj_weight,
                up_proj_weight,
                down_proj_weight,
            });
        }
        println!("\n   All 24 layers successfully loaded into host memory.");

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass for a single token at a given position. Returns logits across vocab.
    pub fn forward_token(
        &self,
        token_id: usize,
        pos: usize,
        cache: &mut KVCache,
        linear: &LinearKernel,
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<f32>, String> {
        let hidden = self.config.hidden_size;
        let mut x = vec![0.0f32; hidden];

        // 1. Embedding lookup
        let emb_offset = token_id * hidden;
        x.copy_from_slice(&self.embed_tokens[emb_offset..emb_offset + hidden]);

        // 2. Transformer layers
        for (l, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();

            // Input LayerNorm
            let x_norm = rms_norm(&x, &layer.input_layernorm, self.config.rms_norm_eps);

            // Self-Attention projections
            let q_dim = self.config.num_attention_heads * self.config.head_dim(); // 14 * 64 = 896
            let kv_dim = self.config.num_key_value_heads * self.config.head_dim(); // 2 * 64 = 128

            let mut q = linear.forward(stream, &x_norm, &layer.q_proj_weight, Some(&layer.q_proj_bias), 1, hidden, q_dim)?;
            let mut k = linear.forward(stream, &x_norm, &layer.k_proj_weight, Some(&layer.k_proj_bias), 1, hidden, kv_dim)?;
            let v = linear.forward(stream, &x_norm, &layer.v_proj_weight, Some(&layer.v_proj_bias), 1, hidden, kv_dim)?;

            // Apply RoPE
            apply_rope(&mut q, &mut k, pos, self.config.num_attention_heads, self.config.num_key_value_heads, self.config.head_dim(), self.config.rope_theta);

            // Update KV Cache
            let layer_cache = &mut cache.layers[l];
            layer_cache.k.extend_from_slice(&k);
            layer_cache.v.extend_from_slice(&v);

            let total_seq_len = layer_cache.k.len() / kv_dim;

            // Compute Multi-Head / Grouped-Query Attention
            let num_q_heads = self.config.num_attention_heads;
            let num_kv_heads = self.config.num_key_value_heads;
            let head_dim = self.config.head_dim();
            let heads_per_kv = num_q_heads / num_kv_heads; // 7

            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut attn_out = vec![0.0f32; q_dim];

            for h in 0..num_q_heads {
                let kv_h = h / heads_per_kv;
                let q_head = &q[h * head_dim..(h + 1) * head_dim];

                // Attention scores for this head across all cached positions
                let mut scores = Vec::with_capacity(total_seq_len);
                for t in 0..total_seq_len {
                    let k_head = &layer_cache.k[t * kv_dim + kv_h * head_dim..t * kv_dim + (kv_h + 1) * head_dim];
                    let dot: f32 = q_head.iter().zip(k_head.iter()).map(|(&a, &b)| a * b).sum();
                    scores.push(dot * scale);
                }

                // Softmax
                softmax(&mut scores);

                // Weighted sum over V
                let out_head = &mut attn_out[h * head_dim..(h + 1) * head_dim];
                for (t, &score) in scores.iter().enumerate() {
                    let v_head = &layer_cache.v[t * kv_dim + kv_h * head_dim..t * kv_dim + (kv_h + 1) * head_dim];
                    for i in 0..head_dim {
                        out_head[i] += score * v_head[i];
                    }
                }
            }

            // Output projection
            let attn_proj = linear.forward(stream, &attn_out, &layer.o_proj_weight, None, 1, q_dim, hidden)?;

            // Residual connection
            for i in 0..hidden {
                x[i] = residual[i] + attn_proj[i];
            }

            // MLP block
            let mlp_residual = x.clone();
            let mlp_norm = rms_norm(&x, &layer.post_attention_layernorm, self.config.rms_norm_eps);

            let intermediate = self.config.intermediate_size;
            let gate = linear.forward(stream, &mlp_norm, &layer.gate_proj_weight, None, 1, hidden, intermediate)?;
            let up = linear.forward(stream, &mlp_norm, &layer.up_proj_weight, None, 1, hidden, intermediate)?;

            // SwiGLU activation
            let mut act = Vec::with_capacity(intermediate);
            for i in 0..intermediate {
                act.push(silu(gate[i]) * up[i]);
            }

            // Down projection
            let mlp_out = linear.forward(stream, &act, &layer.down_proj_weight, None, 1, intermediate, hidden)?;

            for i in 0..hidden {
                x[i] = mlp_residual[i] + mlp_out[i];
            }
        }

        // Final RMSNorm
        let final_norm = rms_norm(&x, &self.norm, self.config.rms_norm_eps);

        // LM Head (tied with embeddings: [vocab_size, hidden])
        linear.forward(stream, &final_norm, &self.embed_tokens, None, 1, hidden, self.config.vocab_size)
    }
}

#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    let inv_std = 1.0 / (ss + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(&xi, &wi)| xi * inv_std * wi).collect()
}

fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
) {
    let half_dim = head_dim / 2;
    for h in 0..num_q_heads {
        let q_head = &mut q[h * head_dim..(h + 1) * head_dim];
        for i in 0..half_dim {
            let freq = 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32);
            let val = pos as f32 * freq;
            let cos = val.cos();
            let sin = val.sin();

            let x0 = q_head[i];
            let x1 = q_head[i + half_dim];
            q_head[i] = x0 * cos - x1 * sin;
            q_head[i + half_dim] = x1 * cos + x0 * sin;
        }
    }

    for h in 0..num_kv_heads {
        let k_head = &mut k[h * head_dim..(h + 1) * head_dim];
        for i in 0..half_dim {
            let freq = 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32);
            let val = pos as f32 * freq;
            let cos = val.cos();
            let sin = val.sin();

            let x0 = k_head[i];
            let x1 = k_head[i + half_dim];
            k_head[i] = x0 * cos - x1 * sin;
            k_head[i + half_dim] = x1 * cos + x0 * sin;
        }
    }
}

fn softmax(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let max_val = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum += *s;
    }
    let inv_sum = 1.0 / sum;
    for s in scores.iter_mut() {
        *s *= inv_sum;
    }
}
