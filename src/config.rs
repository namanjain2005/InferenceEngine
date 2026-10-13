use serde::Deserialize;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
/// Shared decoder dimensions consumed by model backends. Architecture-specific
/// config parsing can populate this type without coupling inference to Qwen's
/// config filename or naming convention.
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
    #[serde(default)]
    pub bos_token_id: usize,
    #[serde(default)]
    pub eos_token_id: usize,
}

impl ModelConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file =
            File::open(path.as_ref()).map_err(|e| format!("Failed to open config.json: {}", e))?;
        serde_json::from_reader(file).map_err(|e| format!("Failed to parse config.json: {}", e))
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.vocab_size == 0
        {
            return Err("Model dimensions must all be non-zero".into());
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(format!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                self.hidden_size, self.num_attention_heads
            ));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(format!(
                "num_attention_heads {} is not divisible by num_key_value_heads {}",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if self.head_dim() % 2 != 0 {
            return Err(format!(
                "head dimension {} must be even for RoPE",
                self.head_dim()
            ));
        }
        Ok(())
    }
}
