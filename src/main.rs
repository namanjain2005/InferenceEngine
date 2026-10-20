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
use kernels::ops::{ArgmaxKernel, AttentionKernel, RmsNormKernel, SwiGluKernel, VectorKernels};
use kernels::{LinearOp, WeightFormat};
use model::{validate_qwen2_checkpoint, ModelBackend, Qwen2Model};
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

    /// Resident projection weight format: f32, f16, or symmetric row-wise int8.
    #[arg(long, default_value = "f32")]
    weight_format: String,

    /// Run static independent-request GPU decode batches using the base prompt suite.
    /// Example: --request-batch-sizes 1,2,5,8
    #[arg(long, value_delimiter = ',')]
    request_batch_sizes: Vec<usize>,
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
        let selected =
            select_gpu(None, &gpus).map_err(|e| format!("GPU selection error: {}", e))?;
        print_gpu_table(&gpus, selected);
        selected
    };

    // 2. Initialize CUDA device
    println!("[2/5] Initializing CUDA Context on GPU {}...", selected_gpu);
    let (cuda_ctx, cuda_stream) = init_cuda_device(selected_gpu)?;
    let weight_format = match args.weight_format.to_lowercase().as_str() {
        "f32" | "fp32" => WeightFormat::F32,
        "f16" | "fp16" => WeightFormat::F16,
        "i8" | "int8" => WeightFormat::I8,
        other => {
            return Err(
                format!("Unknown --weight-format '{other}': expected f32, f16, or int8").into(),
            );
        }
    };
    let linear_custom = LinearKernel::new_with_format(&cuda_ctx, weight_format)?;
    let linear_cublas =
        CublasLinear::new_with_format(&cuda_ctx, cuda_stream.clone(), weight_format)?;
    let backend = args.linear.to_lowercase();
    if !["custom", "cublas", "both"].contains(&backend.as_str()) {
        return Err(format!(
            "Unknown --linear '{}': expected custom, cublas, or both",
            args.linear
        )
        .into());
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
        backends
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "   -> Projection weight format: {:?} (FP32 activations and accumulation)",
        weight_format
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
    let projection_elements = projection_weight_elements(&config);
    let non_projection_elements = (parameter_count as u128).saturating_sub(projection_elements);
    let bytes_per_projection_weight = match weight_format {
        WeightFormat::F32 => 4u128,
        WeightFormat::F16 => 2,
        WeightFormat::I8 => 1,
    };
    let scale_rows = if weight_format == WeightFormat::I8 {
        config.num_hidden_layers as u128
            * (config.num_attention_heads as u128 * config.head_dim() as u128
                + 2 * config.num_key_value_heads as u128 * config.head_dim() as u128
                + 2 * config.hidden_size as u128
                + 2 * config.intermediate_size as u128)
            + if config.tie_word_embeddings {
                0
            } else {
                config.vocab_size as u128
            }
    } else {
        0
    };
    let resident_weight_bytes = projection_elements * bytes_per_projection_weight
        + non_projection_elements * 4
        + scale_rows * 4;
    println!(
        "   -> Safetensors opened: {} tensors in header.",
        st.tensor_count()
    );
    println!(
        "   -> Validated {parameter_count} parameters ({:.2} GiB source weights; {:.2} GiB estimated resident weights as {:?}).",
        checkpoint_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        resident_weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        weight_format
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
            let mut cache = model.new_cache(&cuda_stream)?;
            // Token loop warmup (last-logits form, same as the timed baseline).
            for (pos, &tok) in tiny.iter().enumerate() {
                if pos + 1 < tiny.len() {
                    model.forward_token_hidden(
                        tok,
                        pos,
                        &mut cache,
                        linear,
                        &rmsnorm,
                        &swiglu,
                        &attention,
                        &vector,
                        &cuda_stream,
                    )?;
                } else {
                    model.forward_token(
                        tok,
                        pos,
                        &mut cache,
                        linear,
                        &rmsnorm,
                        &swiglu,
                        &attention,
                        &vector,
                        &cuda_stream,
                    )?;
                }
            }
            let mut cache_b = model.new_cache(&cuda_stream)?;
            let _ = model.forward_prefill(
                &tiny,
                0,
                &mut cache_b,
                linear,
                &rmsnorm,
                &swiglu,
                &attention,
                &vector,
                &cuda_stream,
                false,
            )?;
            let mut cache_g = model.new_cache(&cuda_stream)?;
            let _ = model.forward_prefill(
                &tiny,
                0,
                &mut cache_g,
                linear,
                &rmsnorm,
                &swiglu,
                &attention,
                &vector,
                &cuda_stream,
                true,
            )?;
            println!("   -> Warmup done for backend {name}.");
        }
        cuda_stream
            .synchronize()
            .map_err(|e| format!("Warmup synchronize failed: {e}"))?;
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
        return Err(format!(
            "Unknown --prefill '{}': expected gemv-loop, batched-gemv, or gemm",
            args.prefill
        )
        .into());
    }
    for (index, (label, prompt)) in prompts.iter().enumerate() {
        println!(
            "\n--- Prompt {}/{}: {} ---",
            index + 1,
            prompts.len(),
            label
        );
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

    if !args.request_batch_sizes.is_empty() {
        run_request_batch_suite(
            &prompts,
            &args.request_batch_sizes,
            &model,
            &backends,
            &tokenizer,
            &argmax,
            &rmsnorm,
            &swiglu,
            &attention,
            &vector,
            &cuda_stream,
            args.max_tokens,
            &prefill,
        )?;
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
fn run_request_batch_suite(
    prompts: &[(String, String)],
    requested_sizes: &[usize],
    model: &Qwen2Model,
    backends: &[(&str, &dyn LinearOp)],
    tokenizer: &Tokenizer,
    argmax: &ArgmaxKernel,
    rmsnorm: &RmsNormKernel,
    swiglu: &SwiGluKernel,
    attention: &AttentionKernel,
    vector: &VectorKernels,
    stream: &Arc<CudaStream>,
    max_tokens: usize,
    prefill_mode: &str,
) -> Result<(), String> {
    println!("\n[static request batch suite] Requests arrive together; every prompt owns an independent KV cache.");
    println!("Prefill uses the existing per-request GEMM path; decode projections/MLP/logits use a GPU request batch; attention launches against each request's own cache.");
    let mut encoded = Vec::with_capacity(prompts.len());
    for (label, prompt) in prompts {
        encoded.push((label.clone(), tokenize_prompt(tokenizer, prompt)?));
    }
    let mut reference_cache = std::collections::HashMap::<(String, String), Vec<u32>>::new();

    for &size in requested_sizes {
        if size == 0 {
            return Err("Request batch sizes must be non-zero".into());
        }
        if size > encoded.len() {
            println!(
                "Skipping request batch size {size}: only {} prompts are available",
                encoded.len()
            );
            continue;
        }
        let mut starts = vec![0];
        if encoded.len() > size && encoded.len() - size != 0 {
            starts.push(encoded.len() - size);
        }
        starts.sort_unstable();
        starts.dedup();
        for start_index in starts {
            let group = &encoded[start_index..start_index + size];
            let case = if size == 1 {
                "single-request"
            } else if group.iter().any(|(_, t)| t.len() >= 250)
                && group.iter().any(|(_, t)| t.len() < 80)
            {
                "mixed-short-long"
            } else {
                "prompt-window"
            };
            for &(backend_name, linear) in backends {
                println!(
                    "\n--- GPU request batch: size={size}, case={case}, backend={backend_name} ---"
                );
                println!(
                    "Requests: {}",
                    group
                        .iter()
                        .map(|(name, _)| name.as_str())
                        .collect::<Vec<_>>()
                        .join(" | ")
                );
                // Independent, single-request outputs are the correctness reference.
                let mut references = Vec::with_capacity(size);
                for (label, tokens) in group {
                    let key = (backend_name.to_string(), label.clone());
                    if !reference_cache.contains_key(&key) {
                        let baseline = run_backend(
                            backend_name,
                            label,
                            model,
                            linear,
                            tokenizer,
                            argmax,
                            rmsnorm,
                            swiglu,
                            attention,
                            vector,
                            stream,
                            tokens,
                            max_tokens,
                            prefill_mode,
                            false,
                            1,
                        )?;
                        reference_cache.insert(key.clone(), baseline.tokens);
                    }
                    references.push(
                        reference_cache
                            .get(&key)
                            .expect("reference inserted")
                            .clone(),
                    );
                }

                let mut caches = Vec::with_capacity(size);
                let mut positions = Vec::with_capacity(size);
                let mut next_tokens = Vec::with_capacity(size);
                let mut prefill_ms = Vec::with_capacity(size);
                for (_, prompt_tokens) in group {
                    let capacity = prompt_tokens.len().saturating_add(max_tokens).max(1);
                    let mut cache = model.new_cache_with_capacity(stream, capacity)?;
                    let begin = std::time::Instant::now();
                    let logits = model.forward_prefill(
                        prompt_tokens,
                        0,
                        &mut cache,
                        linear,
                        rmsnorm,
                        swiglu,
                        attention,
                        vector,
                        stream,
                        prefill_mode == "gemm",
                    )?;
                    stream
                        .synchronize()
                        .map_err(|e| format!("Batch prefill synchronization failed: {e}"))?;
                    prefill_ms.push(begin.elapsed().as_secs_f64() * 1000.0);
                    next_tokens.push(select_token(argmax, stream, &logits)?);
                    positions.push(prompt_tokens.len());
                    caches.push(cache);
                }

                let mut generated = vec![Vec::<u32>::with_capacity(max_tokens); size];
                let mut finished = vec![false; size];
                let mut decode_ms = 0.0f64;
                for generated_index in 0..max_tokens {
                    for row in 0..size {
                        let token = next_tokens[row];
                        if !finished[row] {
                            if token == model.config.eos_token_id
                                || token == 151645
                                || token == 151643
                            {
                                finished[row] = true;
                            } else {
                                generated[row].push(token as u32);
                            }
                        }
                    }
                    if generated_index + 1 == max_tokens || finished.iter().all(|&f| f) {
                        break;
                    }
                    // Finished rows remain in the fixed GPU batch with a harmless BOS token;
                    // their outputs are ignored while active rows continue together.
                    let batch_tokens: Vec<usize> = next_tokens
                        .iter()
                        .enumerate()
                        .map(|(row, &token)| {
                            if finished[row] {
                                model.config.bos_token_id
                            } else {
                                token
                            }
                        })
                        .collect();
                    let begin = std::time::Instant::now();
                    let logits = model.forward_decode_batch(
                        &batch_tokens,
                        &positions,
                        &mut caches,
                        linear,
                        rmsnorm,
                        swiglu,
                        attention,
                        vector,
                        stream,
                    )?;
                    let batch_argmax =
                        argmax.forward_batch(stream, &logits, size, model.config.vocab_size)?;
                    next_tokens = stream
                        .clone_dtoh(&batch_argmax)
                        .map_err(|e| format!("Copying batched token IDs failed: {e}"))?
                        .into_iter()
                        .map(|t| {
                            usize::try_from(t).map_err(|_| format!("Invalid batched token ID {t}"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    stream
                        .synchronize()
                        .map_err(|e| format!("Batched decode synchronization failed: {e}"))?;
                    decode_ms += begin.elapsed().as_secs_f64() * 1000.0;
                    for pos in &mut positions {
                        *pos += 1;
                    }
                }
                let prefill_total: f64 = prefill_ms.iter().sum();
                let output_tokens: usize = generated.iter().map(Vec::len).sum();
                let throughput = if decode_ms > 0.0 {
                    output_tokens as f64 * 1000.0 / decode_ms
                } else {
                    0.0
                };
                println!("Batch prefill total (serial prompts): {prefill_total:.2} ms; batched decode: {decode_ms:.2} ms; aggregate decode throughput: {throughput:.2} output tok/s");
                for row in 0..size {
                    let matches = generated[row] == references[row];
                    println!("request={} input_tokens={} prefill_ms={:.2} output_tokens={} identical_to_single={} output={}",
                        group[row].0, group[row].1.len(), prefill_ms[row], generated[row].len(), matches,
                        tokenizer.decode(&generated[row], true).unwrap_or_else(|_| "<decode error>".to_string()));
                    println!("REQUEST_BATCH_RESULT,model_hidden={},layers={},backend={},case={},batch={},request={},input_tokens={},prefill_ms={:.3},batch_decode_ms={:.3},batch_output_tok_s={:.3},output_tokens={},identical={}",
                        model.config.hidden_size, model.config.num_hidden_layers, backend_name, case, size,
                        group[row].0.replace(',', " "), group[row].1.len(), prefill_ms[row], decode_ms,
                        throughput, generated[row].len(), matches);
                    if !matches {
                        return Err(format!(
                            "Static batch output mismatch for request '{}' at batch size {size}",
                            group[row].0
                        ));
                    }
                }
            }
        }
    }
    Ok(())
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
    // Prefill helpers. The token-loop baseline projects logits for the last
    // token only (via forward_token_hidden for the rest), matching the work
    // the batched paths do.
    let prefill_loop =
        |cache: &mut M::Cache| -> Result<(CudaSlice<f32>, std::time::Duration), String> {
            let start = std::time::Instant::now();
            let mut last = None;
            for (pos, &tok) in host_tokens.iter().enumerate() {
                if pos + 1 < host_tokens.len() {
                    model.forward_token_hidden(
                        tok, pos, cache, linear, rmsnorm, swiglu, attention, vector, stream,
                    )?;
                } else {
                    last = Some(model.forward_token(
                        tok, pos, cache, linear, rmsnorm, swiglu, attention, vector, stream,
                    )?);
                }
            }
            stream
                .synchronize()
                .map_err(|e| format!("Prefill loop synchronize failed: {e}"))?;
            Ok((
                last.ok_or("Prompt prefill produced no logits")?,
                start.elapsed(),
            ))
        };
    let prefill_batched = |cache: &mut M::Cache,
                           use_gemm: bool|
     -> Result<(CudaSlice<f32>, std::time::Duration), String> {
        let start = std::time::Instant::now();
        let logits = model.forward_prefill(
            host_tokens,
            0,
            cache,
            linear,
            rmsnorm,
            swiglu,
            attention,
            vector,
            stream,
            use_gemm,
        )?;
        stream
            .synchronize()
            .map_err(|e| format!("Prefill batched synchronize failed: {e}"))?;
        Ok((logits, start.elapsed()))
    };
    let median_of = |mut dts: Vec<std::time::Duration>| -> std::time::Duration {
        dts.sort();
        dts[dts.len() / 2]
    };

    let mut cache;
    let (prefill_time, mut next_token, prefill_logits) = if bench_prefill {
        let reps = bench_repeat;
        // One cache per path, reused for the logits/diff run, the timed reps,
        // and then decode. Prefill rewrites rows `[0, prompt)` in place, so no
        // run needs a fresh allocation and no cache memory is churned.
        let mut cache_loop = model.new_cache(stream)?;
        let mut cache_bgemv = model.new_cache(stream)?;
        let mut cache_gemm = model.new_cache(stream)?;
        let (logits_loop, _) = prefill_loop(&mut cache_loop)?;
        let (logits_bgemv, _) = prefill_batched(&mut cache_bgemv, false)?;
        let (logits_gemm, _) = prefill_batched(&mut cache_gemm, true)?;
        // Timed medians on the same caches, interleaved round-robin so a slow
        // drift of machine state can't systematically favor one path.
        let mut loop_dts = Vec::with_capacity(reps);
        let mut bgemv_dts = Vec::with_capacity(reps);
        let mut gemm_dts = Vec::with_capacity(reps);
        for _ in 0..reps {
            loop_dts.push(prefill_loop(&mut cache_loop)?.1);
            bgemv_dts.push(prefill_batched(&mut cache_bgemv, false)?.1);
            gemm_dts.push(prefill_batched(&mut cache_gemm, true)?.1);
        }
        let (dt_loop, dt_bgemv, dt_gemm) = (
            median_of(loop_dts),
            median_of(bgemv_dts),
            median_of(gemm_dts),
        );
        let host_loop = stream
            .clone_dtoh(&logits_loop)
            .map_err(|e| format!("Prefill benchmark D2H failed: {e}"))?;
        let host_bgemv = stream
            .clone_dtoh(&logits_bgemv)
            .map_err(|e| format!("Prefill benchmark D2H failed: {e}"))?;
        let host_gemm = stream
            .clone_dtoh(&logits_gemm)
            .map_err(|e| format!("Prefill benchmark D2H failed: {e}"))?;
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
        cache = model.new_cache(stream)?;
        match prefill_mode {
            "gemv-loop" => {
                let start = std::time::Instant::now();
                let (logits, _) = prefill_loop(&mut cache)?;
                let host = stream
                    .clone_dtoh(&logits)
                    .map_err(|e| format!("Prefill D2H failed: {e}"))?;
                (
                    start.elapsed(),
                    select_token(argmax, stream, &logits)?,
                    host,
                )
            }
            "batched-gemv" => {
                let (logits, dt) = prefill_batched(&mut cache, false)?;
                let host = stream
                    .clone_dtoh(&logits)
                    .map_err(|e| format!("Prefill D2H failed: {e}"))?;
                (dt, select_token(argmax, stream, &logits)?, host)
            }
            _ => {
                let (logits, dt) = prefill_batched(&mut cache, true)?;
                let host = stream
                    .clone_dtoh(&logits)
                    .map_err(|e| format!("Prefill D2H failed: {e}"))?;
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
            speedup_phrase(
                base.prefill_time,
                other.prefill_time,
                &base.name,
                &other.name
            )
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

fn projection_weight_elements(config: &ModelConfig) -> u128 {
    let hidden = config.hidden_size as u128;
    let q = config.num_attention_heads as u128 * config.head_dim() as u128;
    let kv = config.num_key_value_heads as u128 * config.head_dim() as u128;
    let intermediate = config.intermediate_size as u128;
    config.num_hidden_layers as u128
        * (q * hidden
            + 2 * kv * hidden
            + hidden * q
            + 2 * intermediate * hidden
            + hidden * intermediate)
        + if config.tie_word_embeddings {
            0
        } else {
            config.vocab_size as u128 * hidden
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
