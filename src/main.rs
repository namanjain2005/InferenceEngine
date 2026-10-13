mod config;
mod gpu;
mod kernels;
mod model;
mod safetensors;

use clap::Parser;
use config::ModelConfig;
use cudarc::driver::CudaStream;
use gpu::{init_cuda_device, print_gpu_table, query_nvidia_smi, select_gpu};
use kernels::linear::LinearKernel;
use kernels::ops::{ArgmaxKernel, AttentionKernel, RmsNormKernel, SwiGluKernel, VectorKernels};
use model::{ModelBackend, Qwen2Model, validate_qwen2_checkpoint};
use safetensors::SafeTensors;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

const BASE_PROMPT_TEXT: &str = include_str!("../BASE_PROMPTS.txt");

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Minimal causal language model inference engine in Rust + CUDA"
)]
struct Args {
    /// Target GPU device index (defaults to auto-selecting highest free VRAM)
    #[arg(long)]
    gpu: Option<usize>,

    /// Path to model directory containing config.json and safetensors weights
    #[arg(
        long,
        default_value = "/home/23uec572/Naman_jain/InferenceEngine/models/Qwen2.5-0.5B-Instruct"
    )]
    model_dir: PathBuf,

    /// Validate config and checkpoint tensor shapes without initializing CUDA
    #[arg(long)]
    check_model: bool,

    /// Run one custom prompt instead of the built-in prompt suite
    #[arg(long)]
    prompt: Option<String>,

    /// Maximum tokens to generate
    #[arg(long, default_value_t = 48)]
    max_tokens: usize,

    /// Prefill linear path: gemv-loop (baseline, token-at-a-time),
    /// batched-gemv (one batched GEMV per projection), or
    /// gemm (blocked shared-memory GEMM). Decode always uses GEMV.
    #[arg(long, default_value = "gemm")]
    prefill: String,

    /// Run all three prefill paths on each prompt, verify logits match the
    /// baseline, and print a timing comparison before decoding.
    #[arg(long, default_value_t = false)]
    bench_prefill: bool,

    /// Repetitions per prefill path in bench mode; the median is reported.
    /// Catches machine noise on shared GPUs.
    #[arg(long, default_value_t = 3)]
    bench_repeat: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!(" Causal LM Inference Engine (Rust + CUDA)");
    println!("============================================================");

    let args = Args::parse();

    if args.check_model {
        let config = ModelConfig::from_file(args.model_dir.join("config.json"))?;
        config.validate()?;
        let st = SafeTensors::open_model_dir(&args.model_dir)?;
        let (parameters, bytes) = validate_qwen2_checkpoint(&config, &st)?;
        println!("Checkpoint validation passed");
        println!("Architecture: Qwen2 decoder");
        println!(
            "Layers: {}, hidden: {}, intermediate: {}, vocab: {}",
            config.num_hidden_layers,
            config.hidden_size,
            config.intermediate_size,
            config.vocab_size
        );
        println!(
            "Safetensors files: {}, tensors: {}",
            st.file_count(),
            st.tensor_count()
        );
        println!("Validated parameters: {parameters}");
        println!("Source weight bytes: {} MiB", bytes / (1024 * 1024));
        return Ok(());
    }

    // 1. Query nvidia-smi and select GPU
    println!("\n[1/5] Checking NVIDIA GPUs via nvidia-smi...");
    let gpus = query_nvidia_smi().map_err(|e| format!("Failed to query GPUs: {}", e))?;
    let selected_gpu =
        select_gpu(args.gpu, &gpus).map_err(|e| format!("GPU selection error: {}", e))?;
    print_gpu_table(&gpus, selected_gpu);

    // 2. Initialize CUDA device
    println!("[2/5] Initializing CUDA Context on GPU {}...", selected_gpu);
    let (cuda_ctx, cuda_stream) = init_cuda_device(selected_gpu)?;
    let linear = LinearKernel::new(&cuda_ctx)?;
    let rmsnorm = RmsNormKernel::new(&cuda_ctx)?;
    let swiglu = SwiGluKernel::new(&cuda_ctx)?;
    let attention = AttentionKernel::new(&cuda_ctx)?;
    let vector = VectorKernels::new(&cuda_ctx)?;
    let argmax = ArgmaxKernel::new(&cuda_ctx)?;

    // Test a quick device allocation to verify context is live
    let test_buf = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let dev_test = cuda_stream.clone_htod(&test_buf)?;
    let host_test = cuda_stream.clone_dtoh(&dev_test)?;
    assert_eq!(test_buf, host_test);
    println!(
        "   -> GPU {} VRAM allocation and memory transfer verified.",
        selected_gpu
    );

    // 3. Load Tokenizer
    println!(
        "\n[3/5] Loading Tokenizer from: {}",
        args.model_dir.display()
    );
    let tokenizer_path = args.model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
    println!("   -> Tokenizer successfully loaded.");

    // 4. Load Model Config and Safetensors
    println!("\n[4/5] Loading Model Config & Safetensors weights...");
    let config_path = args.model_dir.join("config.json");
    let config = ModelConfig::from_file(&config_path)?;
    config.validate()?;
    println!(
        "   -> Config: {} layers, hidden={}, intermediate={}, {} q_heads, {} kv_heads (GQA)",
        config.num_hidden_layers,
        config.hidden_size,
        config.intermediate_size,
        config.num_attention_heads,
        config.num_key_value_heads
    );

    let mut st = SafeTensors::open_model_dir(&args.model_dir)?;
    let (parameter_count, checkpoint_bytes) = validate_qwen2_checkpoint(&config, &st)?;
    println!(
        "   -> Safetensors opened: {} tensors in header.",
        st.tensor_count()
    );
    println!(
        "   -> Validated {parameter_count} parameters ({:.2} GiB source weights; {:.2} GiB as f32 on GPU).",
        checkpoint_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        parameter_count as f64 * 4.0 / (1024.0 * 1024.0 * 1024.0)
    );

    let start_load = std::time::Instant::now();
    let model = Qwen2Model::load(&mut st, config.clone(), &linear, &cuda_stream)?;
    println!("   -> Model loaded in {:.2?}", start_load.elapsed());

    // Warmup: run every prefill path once on throwaway caches so timed runs
    // don't pay first-launch clock/cache ramp. Decode still uses GEMV only.
    {
        let mut cache = model.new_cache(&cuda_stream, 4)?;
        let tiny = vec![config.bos_token_id, config.bos_token_id];
        // Token loop warmup (last-logits form, same as the timed baseline).
        for (pos, &tok) in tiny.iter().enumerate() {
            if pos + 1 < tiny.len() {
                model.forward_token_hidden(tok, pos, &mut cache, &linear, &rmsnorm, &swiglu, &attention, &vector, &cuda_stream)?;
            } else {
                model.forward_token(tok, pos, &mut cache, &linear, &rmsnorm, &swiglu, &attention, &vector, &cuda_stream)?;
            }
        }
        let mut cache_b = model.new_cache(&cuda_stream, 4)?;
        let _ = model.forward_prefill(&tiny, 0, &mut cache_b, &linear, &rmsnorm, &swiglu, &attention, &vector, &cuda_stream, false)?;
        let mut cache_g = model.new_cache(&cuda_stream, 4)?;
        let _ = model.forward_prefill(&tiny, 0, &mut cache_g, &linear, &rmsnorm, &swiglu, &attention, &vector, &cuda_stream, true)?;
        cuda_stream.synchronize().map_err(|e| format!("Warmup synchronize failed: {e}"))?;
        println!("   -> Warmup (all prefill paths) done.");
    }

    // 5. Run the full built-in suite unless a one-off prompt was supplied.
    let prompts = match args.prompt {
        Some(prompt) => vec![("Custom prompt".to_string(), prompt)],
        None => base_prompts(),
    };
    if prompts.is_empty() {
        return Err("No prompts found in BASE_PROMPTS.txt".into());
    }

    println!("\n[5/5] Running {} prompt(s)...", prompts.len());
    let prefill = args.prefill.to_lowercase();
    if !["gemv-loop", "batched-gemv", "gemm"].contains(&prefill.as_str()) {
        return Err(format!("Unknown --prefill '{}': expected gemv-loop, batched-gemv, or gemm", args.prefill).into());
    }
    for (index, (label, prompt)) in prompts.iter().enumerate() {
        if let Err(error) = run_prompt(
            index + 1,
            prompts.len(),
            label,
            prompt,
            args.max_tokens,
            &prefill,
            args.bench_prefill,
            args.bench_repeat.max(1),
            &tokenizer,
            &model,
            &argmax,
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
fn run_prompt<M: ModelBackend>(
    index: usize,
    total: usize,
    label: &str,
    prompt: &str,
    max_tokens: usize,
    prefill_mode: &str,
    bench_prefill: bool,
    bench_repeat: usize,
    tokenizer: &Tokenizer,
    model: &M,
    argmax: &ArgmaxKernel,
    linear: &LinearKernel,
    rmsnorm: &RmsNormKernel,
    swiglu: &SwiGluKernel,
    attention: &AttentionKernel,
    vector: &VectorKernels,
    stream: &Arc<CudaStream>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = model.config();
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
    let host_tokens: Vec<usize> = input_tokens.iter().map(|&t| t as usize).collect();

    // Prefill helpers. The token-loop baseline projects logits for the last
    // token only (via forward_token_hidden for the rest), matching the work
    // the batched paths do; the previous system computed throwaway logits
    // per token, which the `main`-branch column still measures as-shipped.
    let prefill_loop = |cache: &mut M::Cache| -> Result<(cudarc::driver::CudaSlice<f32>, std::time::Duration), String> {
        let start = std::time::Instant::now();
        let mut last = None;
        for (pos, &tok) in host_tokens.iter().enumerate() {
            if pos + 1 < host_tokens.len() {
                model.forward_token_hidden(tok, pos, cache, linear, rmsnorm, swiglu, attention, vector, stream)?;
            } else {
                last = Some(model.forward_token(tok, pos, cache, linear, rmsnorm, swiglu, attention, vector, stream)?);
            }
        }
        stream.synchronize().map_err(|e| format!("Prefill loop synchronize failed: {e}"))?;
        Ok((last.ok_or("Prompt prefill produced no logits")?, start.elapsed()))
    };
    let prefill_batched = |cache: &mut M::Cache, use_gemm: bool| -> Result<(cudarc::driver::CudaSlice<f32>, std::time::Duration), String> {
        let start = std::time::Instant::now();
        let logits = model.forward_prefill(&host_tokens, 0, cache, linear, rmsnorm, swiglu, attention, vector, stream, use_gemm)?;
        stream.synchronize().map_err(|e| format!("Prefill batched synchronize failed: {e}"))?;
        Ok((logits, start.elapsed()))
    };
    /// Median of `n` timed runs on fresh caches.
    let median_of = |mut dts: Vec<std::time::Duration>| -> std::time::Duration {
        dts.sort();
        dts[dts.len() / 2]
    };

    let mut cache = model.new_cache(stream, max_seq_len)?;
    let (prefill_time, mut next_token) = if bench_prefill {
        let reps = bench_repeat;
        // Kept run per path: logits for the diff check + cache for decode.
        let mut cache_loop = model.new_cache(stream, max_seq_len)?;
        let mut cache_bgemv = model.new_cache(stream, max_seq_len)?;
        let mut cache_gemm = model.new_cache(stream, max_seq_len)?;
        let (logits_loop, _) = prefill_loop(&mut cache_loop)?;
        let (logits_bgemv, _) = prefill_batched(&mut cache_bgemv, false)?;
        let (logits_gemm, _) = prefill_batched(&mut cache_gemm, true)?;
        // Timed medians with fresh caches (interleaved round-robin so a
        // slow drift of machine state can't systematically favor one path).
        let mut loop_dts = Vec::with_capacity(reps);
        let mut bgemv_dts = Vec::with_capacity(reps);
        let mut gemm_dts = Vec::with_capacity(reps);
        for _ in 0..reps {
            let mut c = model.new_cache(stream, max_seq_len)?;
            loop_dts.push(prefill_loop(&mut c)?.1);
            let mut c = model.new_cache(stream, max_seq_len)?;
            bgemv_dts.push(prefill_batched(&mut c, false)?.1);
            let mut c = model.new_cache(stream, max_seq_len)?;
            gemm_dts.push(prefill_batched(&mut c, true)?.1);
        }
        let (dt_loop, dt_bgemv, dt_gemm) = (median_of(loop_dts), median_of(bgemv_dts), median_of(gemm_dts));
        let host_loop = stream.clone_dtoh(&logits_loop).map_err(|e| format!("Prefill benchmark D2H failed: {e}"))?;
        let host_bgemv = stream.clone_dtoh(&logits_bgemv).map_err(|e| format!("Prefill benchmark D2H failed: {e}"))?;
        let host_gemm = stream.clone_dtoh(&logits_gemm).map_err(|e| format!("Prefill benchmark D2H failed: {e}"))?;
        let max_diff = |a: &[f32], b: &[f32]| a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        println!(
            "Prefill benchmark ({} tokens): gemv-loop {:.2?} | batched-gemv {:.2?} ({:.2}x) diff {:.2e} | gemm {:.2?} ({:.2}x) diff {:.2e}",
            host_tokens.len(),
            dt_loop,
            dt_bgemv,
            dt_loop.as_secs_f64() / dt_bgemv.as_secs_f64().max(1e-9),
            max_diff(&host_loop, &host_bgemv),
            dt_gemm,
            dt_loop.as_secs_f64() / dt_gemm.as_secs_f64().max(1e-9),
            max_diff(&host_loop, &host_gemm),
        );
        // Decode from the requested mode's cache so generation matches it.
        let (dt, tok, chosen) = match prefill_mode {
            "gemv-loop" => (dt_loop, select_token(argmax, stream, &logits_loop)?, cache_loop),
            "batched-gemv" => (dt_bgemv, select_token(argmax, stream, &logits_bgemv)?, cache_bgemv),
            _ => (dt_gemm, select_token(argmax, stream, &logits_gemm)?, cache_gemm),
        };
        cache = chosen;
        (dt, tok)
    } else {
        match prefill_mode {
            "gemv-loop" => {
                let start = std::time::Instant::now();
                let (logits, _) = prefill_loop(&mut cache)?;
                (start.elapsed(), select_token(argmax, stream, &logits)?)
            }
            "batched-gemv" => {
                let (logits, dt) = prefill_batched(&mut cache, false)?;
                (dt, select_token(argmax, stream, &logits)?)
            }
            _ => {
                let (logits, dt) = prefill_batched(&mut cache, true)?;
                (dt, select_token(argmax, stream, &logits)?)
            }
        }
    };
    let mut current_pos = host_tokens.len();
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
        next_token = select_token(argmax, stream, &logits)?;
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

fn select_token(
    argmax: &ArgmaxKernel,
    stream: &Arc<CudaStream>,
    logits: &cudarc::driver::CudaSlice<f32>,
) -> Result<usize, String> {
    let device_token = argmax.forward(stream, logits)?;
    let host_token = stream
        .clone_dtoh(&device_token)
        .map_err(|e| format!("Copying selected token to host failed: {e}"))?;
    let token = *host_token
        .first()
        .ok_or_else(|| "Argmax returned no token".to_string())?;
    usize::try_from(token).map_err(|_| format!("Argmax returned an invalid token ID: {token}"))
}
