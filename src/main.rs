mod config;
mod gpu;
mod kernels;
mod model;
mod safetensors;

use clap::Parser;
use config::ModelConfig;
use cudarc::driver::{CudaSlice, CudaStream};
use gpu::{init_cuda_device, print_gpu_table, query_nvidia_smi, select_gpu};
use kernels::cublas_linear::CublasLinear;
use kernels::linear::LinearKernel;
use kernels::LinearOp;
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

    /// Which linear backends to run per prompt: `both` (default; custom and
    /// cuBLAS on the same input, outputs + timings compared), `custom`
    /// (hand-rolled GEMV/GEMM kernels only), or `cublas` (sgemm only).
    #[arg(long, default_value = "both")]
    linear: String,
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

    // 1. Select GPU. An explicit index can be validated by CUDA itself, so
    // don't require nvidia-smi in that case (it may be unavailable in a
    // container even when the CUDA driver is accessible).
    println!("\n[1/5] Selecting NVIDIA GPU...");
    let selected_gpu = if let Some(gpu) = args.gpu {
        println!("Using explicitly requested GPU {}.", gpu);
        gpu
    } else {
        println!("Checking available GPUs via nvidia-smi...");
        let gpus = query_nvidia_smi().map_err(|e| format!("Failed to query GPUs: {}", e))?;
        let selected = select_gpu(None, &gpus)
            .map_err(|e| format!("GPU selection error: {}", e))?;
        print_gpu_table(&gpus, selected);
        selected
    };

    // 2. Initialize CUDA device
    println!("[2/5] Initializing CUDA Context on GPU {}...", selected_gpu);
    let (cuda_ctx, cuda_stream) = init_cuda_device(selected_gpu)?;
    let linear_custom = LinearKernel::new(&cuda_ctx)?;
    let linear_cublas = CublasLinear::new(&cuda_ctx, cuda_stream.clone())?;
    let backend = args.linear.to_lowercase();
    if !["custom", "cublas", "both"].contains(&backend.as_str()) {
        return Err(format!("Unknown --linear '{}': expected custom, cublas, or both", args.linear).into());
    }
    // Default `both`: run the custom kernels and cuBLAS on the same input and
    // compare outputs + timings on every prompt, so the hand-rolled path is
    // always measured head-to-head against cuBLAS. `custom`/`cublas` force a
    // single backend (kept for fusing/experiment work).
    let backends: Vec<(&str, &dyn LinearOp)> = match backend.as_str() {
        "custom" => vec![("custom", &linear_custom)],
        "cublas" => vec![("cublas", &linear_cublas)],
        _ => vec![("custom", &linear_custom), ("cublas", &linear_cublas)],
    };
    println!(
        "   -> Linear backend(s): {}",
        backends.iter().map(|(name, _)| *name).collect::<Vec<_>>().join(", ")
    );
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
    // Weights are uploaded once and shared by every backend (identical
    // row-major layout), so the two paths never duplicate resident weights.
    let model = Qwen2Model::load(&mut st, config.clone(), backends[0].1, &cuda_stream)?;
    println!("   -> Model loaded in {:.2?}", start_load.elapsed());

    // Warmup: run every prefill path once per backend on throwaway caches so
    // timed runs don't pay first-launch clock/cache ramp. Decode uses GEMV.
    {
        let tiny = vec![config.bos_token_id, config.bos_token_id];
        for &(name, linear) in &backends {
            let mut cache = model.new_cache(&cuda_stream, 4)?;
            // Token loop warmup (last-logits form, same as the timed baseline).
            for (pos, &tok) in tiny.iter().enumerate() {
                if pos + 1 < tiny.len() {
                    model.forward_token_hidden(tok, pos, &mut cache, linear, &rmsnorm, &swiglu, &attention, &vector, &cuda_stream)?;
                } else {
                    model.forward_token(tok, pos, &mut cache, linear, &rmsnorm, &swiglu, &attention, &vector, &cuda_stream)?;
                }
            }
            let mut cache_b = model.new_cache(&cuda_stream, 4)?;
            let _ = model.forward_prefill(&tiny, 0, &mut cache_b, linear, &rmsnorm, &swiglu, &attention, &vector, &cuda_stream, false)?;
            let mut cache_g = model.new_cache(&cuda_stream, 4)?;
            let _ = model.forward_prefill(&tiny, 0, &mut cache_g, linear, &rmsnorm, &swiglu, &attention, &vector, &cuda_stream, true)?;
            println!("   -> Warmup done for backend {name}.");
        }
        cuda_stream.synchronize().map_err(|e| format!("Warmup synchronize failed: {e}"))?;
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
        println!("\n--- Prompt {}/{}: {} ---", index + 1, prompts.len(), label);
        let host_tokens = tokenize_prompt(&tokenizer, prompt)?;
        println!("Input prompt:\n{prompt}");
        println!(
            "Input tokens: {} (identical input fed to every backend)",
            host_tokens.len()
        );

        // Run every requested backend on the exact same input tokens, then
        // print the generated outputs, timings, and a head-to-head check.
        let mut outcomes = Vec::with_capacity(backends.len());
        for &(backend_name, linear) in &backends {
            let outcome = run_backend(
                backend_name,
                label,
                &model,
                linear,
                &tokenizer,
                &argmax,
                &rmsnorm,
                &swiglu,
                &attention,
                &vector,
                &cuda_stream,
                &host_tokens,
                args.max_tokens,
                &prefill,
                args.bench_prefill,
                args.bench_repeat.max(1),
            )
            .map_err(|error| {
                format!("Prompt '{label}' failed on backend '{backend_name}': {error}")
            })?;
            outcomes.push(outcome);
        }
        print_backend_comparison(&outcomes);
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

/// One backend's full prefill + decode result for a single prompt. Lives in
/// `main` only: the library (model/kernels) still takes a single `LinearOp`.
struct PromptOutcome {
    name: String,
    tokens: Vec<u32>,
    text: String,
    decode_steps: usize,
    prefill_time: std::time::Duration,
    decode_time: std::time::Duration,
    /// Last-token prefill logits, kept on host for the cross-backend diff.
    prefill_logits: Vec<f32>,
}

/// Tokenize a prompt with the Qwen chat template into the exact token ids
/// every backend will consume.
fn tokenize_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<usize>, String> {
    let formatted_prompt = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    );
    let encoding = tokenizer
        .encode(formatted_prompt.as_str(), false)
        .map_err(|e| format!("Tokenization failed: {e}"))?;
    let ids = encoding.get_ids();
    if ids.is_empty() {
        return Err("Prompt produced no input tokens".to_string());
    }
    Ok(ids.iter().map(|&t| t as usize).collect())
}

/// Run one linear backend over shared `host_tokens`: prefill (using
/// `prefill_mode` / optional micro-bench) then greedy decode. Prints the
/// backend's own output and timing and returns everything needed to compare
/// it against the other backends.
#[allow(clippy::too_many_arguments)]
fn run_backend<M: ModelBackend>(
    backend: &str,
    label: &str,
    model: &M,
    linear: &dyn LinearOp,
    tokenizer: &Tokenizer,
    argmax: &ArgmaxKernel,
    rmsnorm: &RmsNormKernel,
    swiglu: &SwiGluKernel,
    attention: &AttentionKernel,
    vector: &VectorKernels,
    stream: &Arc<CudaStream>,
    host_tokens: &[usize],
    max_tokens: usize,
    prefill_mode: &str,
    bench_prefill: bool,
    bench_repeat: usize,
) -> Result<PromptOutcome, String> {
    let config = model.config();
    let max_seq_len = host_tokens
        .len()
        .checked_add(max_tokens)
        .ok_or("Requested sequence length overflow")?;

    // Prefill helpers. The token-loop baseline projects logits for the last
    // token only (via forward_token_hidden for the rest), matching the work
    // the batched paths do.
    let prefill_loop =
        |cache: &mut M::Cache| -> Result<(CudaSlice<f32>, std::time::Duration), String> {
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
    let prefill_batched =
        |cache: &mut M::Cache, use_gemm: bool| -> Result<(CudaSlice<f32>, std::time::Duration), String> {
            let start = std::time::Instant::now();
            let logits = model.forward_prefill(host_tokens, 0, cache, linear, rmsnorm, swiglu, attention, vector, stream, use_gemm)?;
            stream.synchronize().map_err(|e| format!("Prefill batched synchronize failed: {e}"))?;
            Ok((logits, start.elapsed()))
        };
    let median_of = |mut dts: Vec<std::time::Duration>| -> std::time::Duration {
        dts.sort();
        dts[dts.len() / 2]
    };

    let mut cache = model.new_cache(stream, max_seq_len)?;
    let (prefill_time, mut next_token, prefill_logits) = if bench_prefill {
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
        let (dt_loop, dt_bgemv, dt_gemm) =
            (median_of(loop_dts), median_of(bgemv_dts), median_of(gemm_dts));
        let host_loop = stream.clone_dtoh(&logits_loop).map_err(|e| format!("Prefill benchmark D2H failed: {e}"))?;
        let host_bgemv = stream.clone_dtoh(&logits_bgemv).map_err(|e| format!("Prefill benchmark D2H failed: {e}"))?;
        let host_gemm = stream.clone_dtoh(&logits_gemm).map_err(|e| format!("Prefill benchmark D2H failed: {e}"))?;
        println!(
            "[{backend}] Prefill benchmark ({} tokens): gemv-loop {:.2?} | batched-gemv {:.2?} ({:.2}x) diff {:.2e} | gemm {:.2?} ({:.2}x) diff {:.2e}",
            host_tokens.len(),
            dt_loop,
            dt_bgemv,
            dt_loop.as_secs_f64() / dt_bgemv.as_secs_f64().max(1e-9),
            max_abs_diff(&host_loop, &host_bgemv),
            dt_gemm,
            dt_loop.as_secs_f64() / dt_gemm.as_secs_f64().max(1e-9),
            max_abs_diff(&host_loop, &host_gemm),
        );
        // Decode from the requested mode's cache so generation matches it.
        match prefill_mode {
            "gemv-loop" => {
                let tok = select_token(argmax, stream, &logits_loop)?;
                cache = cache_loop;
                (dt_loop, tok, host_loop)
            }
            "batched-gemv" => {
                let tok = select_token(argmax, stream, &logits_bgemv)?;
                cache = cache_bgemv;
                (dt_bgemv, tok, host_bgemv)
            }
            _ => {
                let tok = select_token(argmax, stream, &logits_gemm)?;
                cache = cache_gemm;
                (dt_gemm, tok, host_gemm)
            }
        }
    } else {
        match prefill_mode {
            "gemv-loop" => {
                let start = std::time::Instant::now();
                let (logits, _) = prefill_loop(&mut cache)?;
                let host = stream.clone_dtoh(&logits).map_err(|e| format!("Prefill D2H failed: {e}"))?;
                (start.elapsed(), select_token(argmax, stream, &logits)?, host)
            }
            "batched-gemv" => {
                let (logits, dt) = prefill_batched(&mut cache, false)?;
                let host = stream.clone_dtoh(&logits).map_err(|e| format!("Prefill D2H failed: {e}"))?;
                (dt, select_token(argmax, stream, &logits)?, host)
            }
            _ => {
                let (logits, dt) = prefill_batched(&mut cache, true)?;
                let host = stream.clone_dtoh(&logits).map_err(|e| format!("Prefill D2H failed: {e}"))?;
                (dt, select_token(argmax, stream, &logits)?, host)
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
    let decode_time = generation_start.elapsed();

    let text = tokenizer
        .decode(&generated_tokens, false)
        .map_err(|e| format!("Output decoding failed: {e}"))?;

    println!("\n[{backend}] {label} -> generated output:\n{text}");
    if text.trim().is_empty() {
        println!("[{backend}] [No text was generated]");
    }
    println!("[{backend}] --- End generated output ---");
    let tps = if decode_time.as_secs_f64() > 0.0 {
        decode_steps as f64 / decode_time.as_secs_f64()
    } else {
        0.0
    };
    println!(
        "[{backend}] Timing: prefill {} tokens in {:.2?}; generated {} tokens with {} decode steps in {:.2?} ({:.2} decode tok/s)",
        host_tokens.len(),
        prefill_time,
        generated_tokens.len(),
        decode_steps,
        decode_time,
        tps
    );

    Ok(PromptOutcome {
        name: backend.to_string(),
        tokens: generated_tokens,
        text,
        decode_steps,
        prefill_time,
        decode_time,
        prefill_logits,
    })
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Finite max-abs-diff plus non-finite counts for each side. `f32::max`
/// silently drops NaN, so a NaN-vs-finite pair would otherwise report a
/// misleading 0.0 diff; this keeps the comparison honest.
fn summarize_logits(a: &[f32], b: &[f32]) -> (f32, usize, usize) {
    let mut max_diff = 0.0f32;
    let mut non_finite_a = 0usize;
    let mut non_finite_b = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() {
            non_finite_a += 1;
        }
        if !y.is_finite() {
            non_finite_b += 1;
        }
        if x.is_finite() && y.is_finite() {
            max_diff = max_diff.max((x - y).abs());
        }
    }
    (max_diff, non_finite_a, non_finite_b)
}

/// "A 1.39x faster" phrasing for whichever of the two is quicker.
fn speedup_phrase(
    a_time: std::time::Duration,
    b_time: std::time::Duration,
    a_name: &str,
    b_name: &str,
) -> String {
    let a = a_time.as_secs_f64();
    let b = b_time.as_secs_f64();
    if a <= 0.0 || b <= 0.0 {
        return "n/a".to_string();
    }
    if a < b {
        format!("{a_name} {:.2}x faster", b / a)
    } else if b < a {
        format!("{b_name} {:.2}x faster", a / b)
    } else {
        "tie".to_string()
    }
}

/// Head-to-head summary over the outcomes produced for one prompt. Always
/// prints a timing table; with more than one backend it also checks that the
/// generated tokens match and reports the cross-backend logits diff.
fn print_backend_comparison(outcomes: &[PromptOutcome]) {
    if outcomes.len() < 2 {
        return;
    }
    println!("\n=== Backend comparison ===");
    println!(
        "{:<8} {:>12} {:>12} {:>14} {:>9}",
        "backend", "prefill", "decode", "decode tok/s", "gen tok"
    );
    for run in outcomes {
        let tps = if run.decode_time.as_secs_f64() > 0.0 {
            run.decode_steps as f64 / run.decode_time.as_secs_f64()
        } else {
            0.0
        };
        println!(
            "{:<8} {:>12} {:>12} {:>14.2} {:>9}",
            run.name,
            format!("{:.2?}", run.prefill_time),
            format!("{:.2?}", run.decode_time),
            tps,
            run.tokens.len()
        );
    }

    let base = &outcomes[0];
    for other in &outcomes[1..] {
        println!("\n-- {} vs {} --", base.name, other.name);
        if base.tokens == other.tokens {
            println!(
                "  generated tokens: IDENTICAL ({}/{})",
                base.tokens.len(),
                other.tokens.len()
            );
        } else {
            let first = base
                .tokens
                .iter()
                .zip(other.tokens.iter())
                .position(|(a, b)| a != b);
            let at = first
                .map(|i| i.to_string())
                .unwrap_or_else(|| "length difference".to_string());
            println!(
                "  generated tokens: DIVERGED at index {at} ({} vs {} tokens)",
                base.tokens.len(),
                other.tokens.len()
            );
        }
        let (max_diff, non_finite_base, non_finite_other) =
            summarize_logits(&base.prefill_logits, &other.prefill_logits);
        println!(
            "  prefill last-token logits: max abs diff (finite) {:.3e} | non-finite: {}={}, {}={}",
            max_diff, base.name, non_finite_base, other.name, non_finite_other
        );
        println!(
            "  decoded text: {}",
            if base.text == other.text {
                "identical".to_string()
            } else {
                "DIFFERENT".to_string()
            }
        );
        println!(
            "  prefill: {} {:.2?} | {} {:.2?}  ->  {}",
            base.name,
            base.prefill_time,
            other.name,
            other.prefill_time,
            speedup_phrase(base.prefill_time, other.prefill_time, &base.name, &other.name)
        );
        println!(
            "  decode:  {} {:.2?} | {} {:.2?}  ->  {}",
            base.name,
            base.decode_time,
            other.name,
            other.decode_time,
            speedup_phrase(base.decode_time, other.decode_time, &base.name, &other.name)
        );
        println!(
            "  end-to-end: {} {:.2?} | {} {:.2?}  ->  {}",
            base.name,
            base.prefill_time + base.decode_time,
            other.name,
            other.prefill_time + other.decode_time,
            speedup_phrase(
                base.prefill_time + base.decode_time,
                other.prefill_time + other.decode_time,
                &base.name,
                &other.name
            )
        );
    }
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
