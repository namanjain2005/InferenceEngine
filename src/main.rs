mod config;
mod gpu;
mod kernels;
mod model;
mod safetensors;

use clap::Parser;
use config::QwenConfig;
use gpu::{init_cuda_device, print_gpu_table, query_nvidia_smi, select_gpu};
use kernels::linear::LinearKernel;
use kernels::ops::{AttentionKernel, RmsNormKernel, SwiGluKernel, VectorKernels};
use model::{KVCache, ModelWeights};
use safetensors::SafeTensors;
use std::path::PathBuf;
use tokenizers::Tokenizer;

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

    /// Text prompt to run inference on
    #[arg(long, default_value = "Hello! Tell me who you are in one sentence.")]
    prompt: String,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 20)]
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

    // 5. Run Inference
    println!("\n[5/5] Running Inference on Prompt:");
    println!("   Prompt: \"{}\"\n", args.prompt);

    // Format prompt with Qwen instruct template
    let formatted_prompt = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        args.prompt
    );

    let encoding = tokenizer.encode(formatted_prompt.as_str(), false)
        .map_err(|e| format!("Tokenization failed: {}", e))?;
    let input_tokens = encoding.get_ids();
    println!("   Input token IDs ({} tokens): {:?}", input_tokens.len(), input_tokens);

    let max_seq_len = input_tokens.len().checked_add(args.max_tokens).ok_or("Requested sequence length overflow")?;
    let kv_dim = config.num_key_value_heads.checked_mul(config.head_dim()).ok_or("KV dimension overflow")?;
    let mut cache = KVCache::new(&cuda_stream, config.num_hidden_layers, max_seq_len, kv_dim)?;
    let mut current_pos = 0;
    let mut next_token = 0;

    print!("\n--- Generated Output ---\n");
    use std::io::Write;
    let _ = std::io::stdout().flush();

    // Prefill: process prompt tokens
    let prefill_start = std::time::Instant::now();
    for &tok in input_tokens {
        let logits = model.forward_token(tok as usize, current_pos, &mut cache, &linear, &rmsnorm, &swiglu, &attention, &vector, &cuda_stream)?;
        current_pos += 1;

        // Argmax on final prompt token to predict first response token
        if current_pos == input_tokens.len() {
            next_token = argmax(&logits);
        }
    }
    let prefill_time = prefill_start.elapsed();

    // Decode first token
    if let Ok(piece) = tokenizer.decode(&[next_token as u32], false) {
        print!("{}", piece);
        let _ = std::io::stdout().flush();
    }

    // Autoregressive generation loop
    let gen_start = std::time::Instant::now();
    let mut gen_count = 1;

    for _ in 1..args.max_tokens {
        if next_token == config.eos_token_id || next_token == 151645 || next_token == 151643 {
            break;
        }

        let logits = model.forward_token(next_token, current_pos, &mut cache, &linear, &rmsnorm, &swiglu, &attention, &vector, &cuda_stream)?;
        current_pos += 1;
        next_token = argmax(&logits);
        gen_count += 1;

        if let Ok(piece) = tokenizer.decode(&[next_token as u32], false) {
            print!("{}", piece);
            let _ = std::io::stdout().flush();
        }
    }
    println!("\n------------------------\n");

    println!(
        "Performance: Prefill ({} tokens) in {:.2?}, Generated {} tokens in {:.2?} ({:.2} tok/s)",
        input_tokens.len(),
        prefill_time,
        gen_count,
        gen_start.elapsed(),
        gen_count as f64 / gen_start.elapsed().as_secs_f64()
    );

    println!("\n============================================================");
    println!(" Inference finished successfully on GPU {}!", selected_gpu);
    println!("============================================================");

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

