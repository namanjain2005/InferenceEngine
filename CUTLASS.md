# CUTLASS dependency

The prefill attention kernel uses the fused multi-head attention example from
NVIDIA CUTLASS. CUTLASS is a Git submodule pinned by this repository to
v3.9.2 (`ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e`). The build compiles the
kernel to SM70 PTX with NVCC. A small build-time overlay preserves the GQA
behavior needed by this wrapper without modifying the upstream submodule.

## Clone the repository

With modern Git, initialize CUTLASS while cloning:

```bash
git clone --recurse-submodules https://github.com/namanjain2005/InferenceEngine.git
```

For older Git versions, `--recursive` is the equivalent option:

```bash
git clone --recursive https://github.com/namanjain2005/InferenceEngine.git
```

If the repository is already cloned, or the submodule directory is empty, run
this from the repository root:

```bash
git submodule update --init --recursive
```

## Pull project changes

Pull the project commit, then check out the CUTLASS commit recorded by it:

```bash
git pull
git submodule update --init --recursive
```

The submodule is pinned; pulling the project does not float CUTLASS to its
latest upstream version. Verify the checked-out commit with:

```bash
git submodule status
```

## Intentionally update CUTLASS

Only update the submodule when changing the dependency version deliberately.
Fetch and check out the desired upstream tag, then commit the new submodule
pointer together with any compatibility changes:

```bash
cd vendor/cutlass
git fetch --tags
git checkout v3.9.2
cd ../..
git add vendor/cutlass
git commit -m "Update pinned CUTLASS version"
```

Replace `v3.9.2` with the reviewed target tag when upgrading. After pulling a
commit that updates the pointer, run `git submodule update --init --recursive`
to check out that exact version.

## Build and run the A/B benchmark

Build-time PTX compilation requires `nvcc`. The committed PTX target is SM70,
used by the Tesla V100 runs below. Model checkpoints are local fixtures and
are not included in the Git repository; see [MODEL_CHECKPOINTS.md](MODEL_CHECKPOINTS.md)
for model setup.

From the repository root, compile and run the complete Rust target suite:

```bash
cargo build --release
cargo test --all-targets
```

The default prompt source is `BASE_PROMPTS.txt`. Each command below runs all
12 prompts with 16 generated tokens, FP16 projection weights, FP16 attention
and KV cache, and the same prompt input for each selected implementation.

Run the 1.5B and 3B cases with both linear backends and both attention paths:

```bash
cargo run --release -- --gpu 0 \
  --model-dir models/Qwen2.5-1.5B-Instruct \
  --weight-format fp16 --attention-format fp16 \
  --prefill-attention both --max-tokens 16 --linear both

cargo run --release -- --gpu 0 \
  --model-dir models/Qwen2.5-3B-Instruct \
  --weight-format fp16 --attention-format fp16 \
  --prefill-attention both --max-tokens 16 --linear both
```

For 7B, run each attention path separately with the custom linear backend to
avoid the shared-VRAM pressure observed when switching paths in one process:

```bash
cargo run --release -- --gpu 0 \
  --model-dir models/Qwen2.5-7B-Instruct \
  --weight-format fp16 --attention-format fp16 \
  --prefill-attention flash --max-tokens 16 --linear custom

cargo run --release -- --gpu 0 \
  --model-dir models/Qwen2.5-7B-Instruct \
  --weight-format fp16 --attention-format fp16 \
  --prefill-attention cutlass --max-tokens 16 --linear custom
```

For exact 1,024 / 4,096 / 8,192-token contexts, run one untimed warmup and
three synchronized prefill timings per path; the program reports each median
and compares final logits/token IDs. The 1.5B command is:

```bash
cargo run --release -- --gpu 0 \
  --model-dir models/Qwen2.5-1.5B-Instruct \
  --weight-format fp16 --attention-format fp16 \
  --prefill-attention both --bench-prefill --bench-repeat 3 \
  --prefill gemm --max-tokens 1 --linear cublas \
  --benchmark-contexts 1024,4096,8192
```

Repeat with `models/Qwen2.5-3B-Instruct` and `--linear cublas`. For 7B, use
`models/Qwen2.5-7B-Instruct` and `--linear custom`; the 8K cuBLAS case ran out
of memory in the recorded V100 environment. Context lengths over the default
2,048-token KV capacity exercise cache growth. `--max-tokens 1` limits the
comparison to prefill and avoids timing decode.

The original measurements used a shared Tesla V100-SXM2-32GB with CUDA 12.4
and driver 550.144.03. Other GPU load and model fixtures affect absolute times.
The full prompt and long-context logs, per-prompt CSV, and OOM attempts are
linked in [BENCHMARK.md](BENCHMARK.md).
