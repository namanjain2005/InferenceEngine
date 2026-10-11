mod config;
mod gpu;
mod kernels;
mod model;
mod safetensors;

use clap::Parser;
use config::QwenConfig;
use cudarc::driver::CudaStream;
use gpu::{init_cuda_device, print_gpu_table, query_nvidia_smi, select_gpu};
use kernels::linear::LinearKernel;
use kernels::ops::{AttentionKernel, RmsNormKernel, SwiGluKernel, VectorKernels};
use model::{KVCache, ModelWeights};
use safetensors::SafeTensors;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

const BASE_PROMPT_TEXT: &str = include_str!("../BASE_PROMPTS.txt");

#[derive(Parser, Debug)]
#[command(author, version, about = "Minimal Qwen2.5-0.5B-Instruct Inference Engine in Rust from Scratch")]
struct Args {
    /// Target GPU device index (defaults to auto-selecting highest free VRAM)
    #[arg(long)]
    gpu: Option<usize>,

    /// Path to model directory containing model.safetensors and config.json
    #[arg(
        long,
        default_value = "/home/23uec572/Naman_jain/InferenceEngine/models/Qwen2.5-0.5B-Instruct"
    )]
    model_dir: PathBuf,

    /// Run one custom prompt instead of the built-in prompt suite
    #[arg(long)]
    prompt: Option<String>,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 48)]
    max_tokens: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" Qwen 500M Inference Engine (From-Scratch Rust + CUDA)");
    println!("============================================================");

    let args = Args::parse();

    // 1. Query nvidia-smi and select GPU
    println!("\n[1/5] Checking NVIDIA GPUs via nvidia-smi...");
    let gpus = query_nvidia_smi().map_err(|e| format!("Failed to query GPUs: {}", e))?;
    let selected_gpu = select_gpu(args.gpu, &gpus).map_err(|e| format!("GPU selection error: {}", e))?;
    print_gpu_table(&gpus, selected_gpu);

    // 2. Initialize CUDA device
    println!("[2/5] Initializing CUDA Context on GPU {}...", selected_gpu);
    let (cuda_ctx, cuda_stream) = init_cuda_device(selected_gpu)?;
    let linear = LinearKernel::new(&cuda_ctx)?;
    let rmsnorm = RmsNormKernel::new(&cuda_ctx)?;
    let swiglu = SwiGluKernel::new(&cuda_ctx)?;
    let attention = AttentionKernel::new(&cuda_ctx)?;
    let vector = VectorKernels::new(&cuda_ctx)?;

    // Test a quick device allocation to verify context is live
    let test_buf = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let dev_test = cuda_stream.clone_htod(&test_buf)?;
    let host_test = cuda_stream.clone_dtoh(&dev_test)?;
    assert_eq!(test_buf, host_test);
    println!("   -> GPU {} VRAM allocation and memory transfer verified.", selected_gpu);

    // 3. Load Tokenizer
    println!("\n[3/5] Loading Tokenizer from: {}", args.model_dir.display());
    let tokenizer_path = args.model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
    println!("   -> Tokenizer successfully loaded.");

    // 4. Load Model Config and Safetensors
    println!("\n[4/5] Loading Model Config & Safetensors weights...");
    let config_path = args.model_dir.join("config.json");
    let config = QwenConfig::from_file(&config_path)?;
    println!(
        "   -> Config: {} layers, hidden={}, intermediate={}, {} q_heads, {} kv_heads (GQA)",
        config.num_hidden_layers,
        config.hidden_size,
        config.intermediate_size,
        config.num_attention_heads,
        config.num_key_value_heads
    );

    let safetensors_path = args.model_dir.join("model.safetensors");
    let mut st = SafeTensors::open(&safetensors_path)?;
    println!("   -> Safetensors opened: {} tensors in header.", st.tensors.len());

    let start_load = std::time::Instant::now();
    let model = ModelWeights::load(&mut st, config.clone(), &linear, &cuda_stream)?;
    println!("   -> Model loaded in {:.2?}", start_load.elapsed());

    // 5. Run the full built-in suite unless a one-off prompt was supplied.
    let prompts = match args.prompt {
        Some(prompt) => vec![("Custom prompt".to_string(), prompt)],
        None => base_prompts(),
    };
    if prompts.is_empty() {
        return Err("No prompts found in BASE_PROMPTS.txt".into());
    }

    println!("\n[5/5] Running {} prompt(s)...", prompts.len());
    for (index, (label, prompt)) in prompts.iter().enumerate() {
        if let Err(error) = run_prompt(
            index + 1,
            prompts.len(),
            label,
            prompt,
            args.max_tokens,
            &tokenizer,
            &config,
            &model,
            &linear,
            &rmsnorm,
            &swiglu,
            &attention,
            &vector,
            &cuda_stream,
        ) {
            return Err(format!("Prompt '{label}' failed: {error}").into());
        }
    }

    println!("\n============================================================");
    println!(" Inference finished successfully on GPU {}!", selected_gpu);
    println!("============================================================");

    Ok(())
}

fn base_prompts() -> Vec<(String, String)> {
    let mut prompts = Vec::new();
    let mut label: Option<String> = None;
    let mut body = String::new();

    for line in BASE_PROMPT_TEXT.lines() {
        let trimmed = line.trim();
        if let Some(next_label) = trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            if let Some(previous_label) = label.replace(next_label.to_string()) {
                let prompt = body.trim().to_string();
                if !prompt.is_empty() {
                    prompts.push((previous_label, prompt));
                }
            }
            body.clear();
        } else if label.is_some() {
            body.push_str(line);
            body.push('\n');
        }
    }

    if let Some(final_label) = label {
        let prompt = body.trim().to_string();
        if !prompt.is_empty() {
            prompts.push((final_label, prompt));
        }
    }

    prompts
}

#[allow(clippy::too_many_arguments)]
fn run_prompt(
    index: usize,
    total: usize,
    label: &str,
    prompt: &str,
    max_tokens: usize,
    tokenizer: &Tokenizer,
    config: &QwenConfig,
    model: &ModelWeights,
    linear: &LinearKernel,
    rmsnorm: &RmsNormKernel,
    swiglu: &SwiGluKernel,
    attention: &AttentionKernel,
    vector: &VectorKernels,
    stream: &Arc<CudaStream>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Prompt {index}/{total}: {label} ---");

    let formatted_prompt = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    );
    let encoding = tokenizer
        .encode(formatted_prompt.as_str(), false)
        .map_err(|e| format!("Tokenization failed: {e}"))?;
    let input_tokens = encoding.get_ids();
    if input_tokens.is_empty() {
        return Err(format!("Prompt {label} produced no input tokens").into());
    }
    println!("Input tokens: {}", input_tokens.len());

    let max_seq_len = input_tokens
        .len()
        .checked_add(max_tokens)
        .ok_or("Requested sequence length overflow")?;
    let kv_dim = config
        .num_key_value_heads
        .checked_mul(config.head_dim())
        .ok_or("KV dimension overflow")?;
    let mut cache = KVCache::new(stream, config.num_hidden_layers, max_seq_len, kv_dim)?;
    let mut current_pos = 0;
    let mut first_token = None;

    let prefill_start = std::time::Instant::now();
    for &token in input_tokens {
        let logits = model.forward_token(
            token as usize,
            current_pos,
            &mut cache,
            linear,
            rmsnorm,
            swiglu,
            attention,
            vector,
            stream,
        )?;
        current_pos += 1;
        first_token = Some(argmax(&logits));
    }
    let prefill_time = prefill_start.elapsed();

    let mut next_token = first_token.ok_or("Prompt prefill produced no logits")?;
    let mut generated_tokens = Vec::with_capacity(max_tokens);
    let mut decode_steps = 0usize;
    let generation_start = std::time::Instant::now();
    for generated_count in 0..max_tokens {
        if next_token == config.eos_token_id || next_token == 151645 || next_token == 151643 {
            break;
        }
        generated_tokens.push(next_token as u32);
        if generated_count + 1 == max_tokens {
            break;
        }

        let logits = model.forward_token(
            next_token,
            current_pos,
            &mut cache,
            linear,
            rmsnorm,
            swiglu,
            attention,
            vector,
            stream,
        )?;
        decode_steps += 1;
        current_pos += 1;
        next_token = argmax(&logits);
    }
    let generation_time = generation_start.elapsed();

    let response = tokenizer
        .decode(&generated_tokens, false)
        .map_err(|e| format!("Output decoding failed: {e}"))?;
    println!("Generated output:\n{response}");
    if response.trim().is_empty() {
        println!("[No text was generated]");
    }
    println!("--- End generated output ---");
    let generated_count = generated_tokens.len();
    let tokens_per_second = if generation_time.as_secs_f64() > 0.0 {
        decode_steps as f64 / generation_time.as_secs_f64()
    } else {
        0.0
    };
    println!(
        "Timing: prefill {} tokens in {:.2?}; generated {} tokens with {} decode steps in {:.2?} ({:.2} decode tok/s)",
        input_tokens.len(),
        prefill_time,
        generated_count,
        decode_steps,
        generation_time,
        tokens_per_second
    );
    Ok(())
}

fn argmax(slice: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}

