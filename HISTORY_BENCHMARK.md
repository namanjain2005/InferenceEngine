# Git history review and per-case benchmark

Reviewed all refs through `c923c74` (2026-10-19): 39 reachable commits total,
including the 34-commit `main` ancestry and the attention experiment branches.
This file groups the larger changes, records which evidence exists for each
group, and gives the latest complete per-prompt run. `BENCHMARK.md` remains the
source for the historical prefill and backend comparison tables; it has
prompt-by-prompt results for the 0.5B, 1.5B, and 3B models at those milestones.

## Major changes in the history

| Commits | Change | Evidence and per-test status |
|---|---|---|
| `78386ae`, `fbb9d40`, `28aface` | Rust project, first unit check, Qwen2.5-0.5B CPU inference, CUDA setup, safetensors, tokenizer | The original `compute_sum` unit test was later removed; current `cargo test --all-targets` at that historical revision had no Rust tests. Model loading and inference are exercised by the current full prompt suite below. |
| `cdf15cb`, `d9e1c59` | Remove generated model/build artifacts from Git; small safetensors test adjustment | Repository hygiene and test-fixture maintenance; no performance behavior to benchmark. |
| `8bf4586`, `e8dc505`, `663b115`, `ab059f9` | Batched CUDA linear plus attention, RMSNorm, SwiGLU, vector ops; then reduce host/device copies | No archived per-commit timing per prompt. Current all-prompt correctness and timings are below. Later prefill tables quantify the batched GEMV path. |
| `86bb70d`, `8f96270`, `a84415b`, `5ec2ab3`, `c3054ad`, `3c31837` | CUDA launch/synchronization fixes; parallel attention softmax and warp reductions; tiled linear input | These are correctness and kernel performance changes. The current long-context prompt cases 4 and 5 complete with finite logits and matching backends; older per-commit comparisons are not present. |
| `225e05b`, `c888479`, `6ce357c`, `d2dfbb3` | Run the built-in prompt set and improve prompts/output/error reporting | The complete suite is the nine cases in the tables below. All nine pass on current `main`. |
| `e6b6999`, `0e655a7`, `5db359d`, `4a15196`, `214ffee` (attention experiment branches) | Try attention-before-softmax / attention-after-softmax variants, add comparison prompts, run all prompts, tile linear inputs | These branches establish parallel kernel experiments and prompt coverage. No per-branch timing tables were checked in; see `BENCHMARK.md` for the later measured prefill milestones. |
| `c615f10`, `b84fd45`, `678b2ce` | Qwen family backend, sharded checkpoints, GPU argmax, expanded prompt coverage | `--check-model` passes for both available fixtures (1.5B and 3B). The full nine-prompt suite passes on both available models below. The 0.5B fixture is missing locally. |
| `03990c8`, `9c20188`, `bab988f` | Blocked GEMM prefill plus fair warmup, same-work baseline, interleaved median-of-3 | Historical per-prompt measurements are in [BENCHMARK.md](BENCHMARK.md), including 9 prompts per model. At 321 tokens, loop-to-GEMM was 24.1x (0.5B), 16.2x (1.5B), and 13.7x (3B) in the revised tables. The current per-prompt measurements are below. |
| `a942365`, `012db43`, `6522082`, `7ac105d`, `e0bf0b5` | cuBLAS backend, same-prompt comparison, non-finite reporting, GEMV decode | Historical per-prompt custom/cuBLAS timings and generated-token agreement are in [BENCHMARK.md](BENCHMARK.md) for 0.5B/1.5B/3B. Current runs compare both backends on all nine 1.5B and 3B prompts. |
| `0b4f3e3`, `633e30c`, `d7de285` | Pad attention shared memory; fix softmax shared-memory race; document sanitizer and backend findings | `BENCHMARK.md` records the captured-input race check (15/60 NaN-producing launches before the barrier, 0/80 after) and long-prompt regression evidence. Current prompts 4 and 5 pass with zero non-finite logits. |
| `c923c74` | Geometric KV-cache growth and cache reuse in prefill benchmark | A separate 2070-token prompt crossed the 2048-token initial capacity, completed prefill, and exited successfully; details below. |

The history contains small documentation, prompt, and bug-fix commits in
addition to these groups; they are included above rather than reported as
separate performance claims. Timings cannot be attributed to every historical
commit: most commits predate a stable per-prompt harness, and
replaying arbitrary old revisions would mix changes in prompts, kernels,
checkpoints, compiler output, and measurement methods. The per-case historic
comparisons that can be reproduced from checked-in evidence remain in
`BENCHMARK.md`.

## Current full-suite run

Command:

```bash
cargo run --release -- --gpu 4 \
  --model-dir models/Qwen2.5-1.5B-Instruct \
  --max-tokens 16 --bench-prefill --bench-repeat 3 --linear both
```

Environment: Tesla V100-SXM2-32GB, CUDA driver 550.144.03 / CUDA 12.4,
f32, shared GPU 4. GPU activity changed during the run, so these timings are a
snapshot, not an isolated performance claim. The application ran all nine
prompts through both backends; generated token IDs matched on every case and
all reported non-finite counts were zero. The table reports milliseconds.
The CLI warms up, interleaves paths, and reports median-of-three prefill runs.

### Prefill by prompt and backend

| Prompt | Input tokens | Custom loop | Custom batched GEMV | Custom GEMM | cuBLAS loop | cuBLAS batched GEMV | cuBLAS GEMM | Tokens identical |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Tiny context | 33 | 382.64 | 199.87 | 103.15 | 309.85 | 21.75 | 21.74 | 13/13 |
| Short context | 45 | 522.91 | 274.41 | 103.65 | 422.76 | 22.18 | 22.11 | 16/16 |
| Medium context | 154 | 1820 | 957.47 | 145.47 | 1480 | 51.92 | 54.04 | 16/16 |
| Long context | 321 | 3870 | 2010 | 251.91 | 3170 | 92.76 | 93.22 | 16/16 |
| Long context, repeated structure | 282 | 3390 | 1760 | 210.20 | 2760 | 84.92 | 84.53 | 16/16 |
| Multi-step arithmetic | 56 | 651.83 | 343.65 | 104.14 | 528.25 | 22.49 | 22.53 | 16/16 |
| Compare totals | 73 | 851.46 | 450.67 | 114.74 | 689.54 | 31.32 | 31.32 | 16/16 |
| Exact format | 73 | 848.13 | 450.66 | 114.70 | 689.10 | 31.37 | 31.36 | 16/16 |
| Time extraction | 78 | 910.67 | 481.30 | 114.95 | 737.29 | 31.74 | 31.78 | 16/16 |

Values printed in seconds were converted to milliseconds. The custom GEMM
column is also the custom backend's selected prefill duration; the cuBLAS GEMM
column is the cuBLAS backend's selected prefill duration.

### Decode by prompt and backend

| Prompt | Custom decode ms | cuBLAS decode ms | Max abs difference in prefill logits | Non-finite logits (custom/cuBLAS) |
|---|---:|---:|---:|---:|
| Tiny context | 169.57 | 140.17 | 1.029e-4 | 0 / 0 |
| Short context | 196.30 | 162.24 | 1.020e-4 | 0 / 0 |
| Medium context | 201.63 | 167.84 | 6.151e-5 | 0 / 0 |
| Long context | 209.63 | 177.41 | 6.104e-5 | 0 / 0 |
| Long context, repeated structure | 207.88 | 174.80 | 5.245e-5 | 0 / 0 |
| Multi-step arithmetic | 197.18 | 163.38 | 5.722e-5 | 0 / 0 |
| Compare totals | 197.90 | 163.62 | 8.821e-5 | 0 / 0 |
| Exact format | 197.38 | 165.06 | 9.727e-5 | 0 / 0 |
| Time extraction | 198.24 | 164.45 | 1.030e-4 | 0 / 0 |

`max_tokens=16`; the tiny-context case stopped after 13 tokens. Decode times are
per generation, with the output's decode-step count printed by the application.

The same nine-prompt suite also passed on the available 3B checkpoint, using:

```bash
./target/release/Engine --gpu 4 \
  --model-dir models/Qwen2.5-3B-Instruct \
  --max-tokens 16 --linear both
```

This run reports single-pass prompt timings rather than the repeated prefill
microbenchmark. Both backends generated identical tokens for all nine prompts,
with zero non-finite logits. Times are milliseconds; GPU 4 was shared.

| Prompt | Input tokens | Custom prefill | Custom decode | cuBLAS prefill | cuBLAS decode | Tokens identical |
|---|---:|---:|---:|---:|---:|---|
| Tiny context | 33 | 181.83 | 251.75 | 40.96 | 209.89 | 11/11 |
| Short context | 45 | 184.37 | 344.02 | 41.82 | 287.28 | 16/16 |
| Medium context | 154 | 265.69 | 350.48 | 87.66 | 295.37 | 16/16 |
| Long context | 321 | 483.38 | 361.13 | 181.60 | 307.95 | 16/16 |
| Long context, repeated structure | 282 | 397.97 | 359.24 | 150.74 | 303.80 | 16/16 |
| Multi-step arithmetic | 56 | 185.60 | 344.85 | 42.35 | 288.50 | 16/16 |
| Compare totals | 73 | 203.28 | 344.75 | 57.84 | 288.74 | 16/16 |
| Exact format | 73 | 202.87 | 344.89 | 57.82 | 288.32 | 16/16 |
| Time extraction | 78 | 203.15 | 345.61 | 58.48 | 290.07 | 16/16 |

## Other checks

| Check | Result |
|---|---|
| `cargo test --all-targets` | Passed; 0 tests in the binary target and 0 in each of the two example targets. Cargo emitted existing dead-code warnings for `GpuDevice::util_pct` and two `SafeTensors` methods. |
| `--check-model` on `Qwen2.5-1.5B-Instruct` | Passed: 28 layers, 338 tensors, 1,543,714,304 parameters. |
| `--check-model` on `Qwen2.5-3B-Instruct` | Passed: 36 layers, 434 tensors, 3,085,938,688 parameters. |
| KV-cache boundary (`c923c74`) | Passed: 2070 prompt tokens (above the 2048 initial capacity), cuBLAS backend, 741.02 ms prefill, one generated token; process completed successfully on GPU 4. |
| `gemv_check` example | Passed all seven projection shapes; output finite and checked against CPU reference. |
| `decode_bench` example | Passed all twelve projection shapes; GEMV vs SGEMM max difference was 0.00e0 for every shape. Per-shape measurements below. |

The historical revision had no automated Rust unit or integration test
functions. The precision-format change adds two unit tests, listed below. The
nine-prompt GPU run remains the end-to-end inference suite.

The cache boundary command used a deterministic prompt built from 2050
repetitions of `test `, which tokenized to 2070 tokens:

```bash
prompt=''
for ((i=0; i<2050; i++)); do prompt+='test '; done
./target/release/Engine --gpu 4 \
  --model-dir models/Qwen2.5-1.5B-Instruct \
  --max-tokens 1 --linear cublas --prompt "$prompt"
```

This verifies that a prefill longer than the initial capacity grows the cache
instead of failing. It does not measure long-context decode throughput.

## Standalone decode-kernel checks

Commands: `cargo run --release --example gemv_check -- 4` and
`cargo run --release --example decode_bench -- 4 50`. These examples are not
executed by `cargo test`; both were run directly. The microbenchmark warms up
five iterations and measures 50 iterations per path. It is kernel-only with
preallocated outputs. Times are milliseconds per call on the same shared GPU 4.

| Projection shape | cuBLAS GEMV ms | cuBLAS SGEMM M=1 ms | Max output diff |
|---|---:|---:|---:|
| 896 → 896 (0.5B q/o) | 0.0076 | 0.0078 | 0.00e0 |
| 896 → 128 (0.5B k/v) | 0.0075 | 0.0080 | 0.00e0 |
| 896 → 4864 (0.5B gate/up) | 0.0264 | 0.0264 | 0.00e0 |
| 4864 → 896 (0.5B down) | 0.0304 | 0.0303 | 0.00e0 |
| 896 → 151936 (0.5B logits) | 0.6402 | 0.6407 | 0.00e0 |
| 1536 → 1536 (1.5B q/o) | 0.0196 | 0.0191 | 0.00e0 |
| 1536 → 1536 (1.5B k/v) | 0.0178 | 0.0164 | 0.00e0 |
| 1536 → 8960 (1.5B gate/up) | 0.0709 | 0.0711 | 0.00e0 |
| 8960 → 1536 (1.5B down) | 0.0761 | 0.0762 | 0.00e0 |
| 1536 → 151936 (1.5B logits) | 1.0942 | 1.0914 | 0.00e0 |
| 2048 → 11008 (3B gate/up) | 0.1407 | 0.1408 | 0.00e0 |
| 2048 → 151936 (3B logits) | 1.4598 | 1.4550 | 0.00e0 |



## Precision formats and larger checkpoints (2026-10-20)

Implementation commit: `b4c2681` (`feat(inference): add fp16 and int8 weight formats`).

Commit-level summary: add FP16 projection-weight storage and symmetric
per-output-row INT8 weight quantization, retaining FP32 activations and
accumulation in the custom path; support untied `lm_head` weights; add a
checkpoint-size-aware resident-memory estimate. FP16 cuBLAS casts its input
activations to FP16 for `cublasGemmEx` and accumulates/outputs in FP32. The
INT8 cuBLAS path expands each projection to temporary FP32 before GEMV/GEMM,
so it is functionally covered but slow on decode. These are single-pass
end-to-end timings, not the median-of-three `--bench-prefill` measurements in
the historical tables above.

Qwen does not publish exact 8B/13B sizes in the tested family; Qwen2.5-7B and
Qwen2.5-14B were used as the nearest checkpoints. See
[`MODEL_CHECKPOINTS.md`](MODEL_CHECKPOINTS.md) for pinned revisions and
reproducible download commands. The downloaded checkpoint files are present
locally under `models/` (the 14B path points to `/tmp/InferenceEngine-models`
because of the home-volume quota) and are excluded from Git because together
they occupy about 43 GB.

Environment: GPU 1, Tesla V100-SXM2-32GB, CUDA 12.4 / driver 550.144.03,
release build. Every command used `--max-tokens 16 --linear both`; every
completed format/model combination ran all nine prompts from
`BASE_PROMPTS.txt` on both custom and cuBLAS backends. This was one pass per
prompt; GPU load was not controlled between cases. All 11 completed
combinations produced identical custom/cuBLAS token IDs for every prompt, and
zero non-finite logits. Per-prompt prefill/decode timings, token counts, logit
differences, and full captured outputs are in
[`per_prompt_results.csv`](benchmark-results/precision-2026-10-20/per_prompt_results.csv)
and the adjacent raw `.log` files.

Reproduction (substitute `f32`, `fp16`, or `int8` where supported):

```bash
cargo run --release -- --gpu 1 --model-dir models/Qwen2.5-7B-Instruct \
  --max-tokens 16 --linear both --weight-format fp16
cargo run --release -- --gpu 1 --model-dir models/Qwen2.5-14B-Instruct \
  --max-tokens 16 --linear both --weight-format fp16
```

The 7B checkpoint completed the full suite in F32, FP16, and INT8. The 14B
checkpoint completed in FP16 and INT8; FP32 was not attempted because its
estimated 55.02 GiB of weights alone exceeds this GPU's 32 GiB. FP16 14B
estimated 28.96 GiB resident weights and left little free VRAM during loading;
INT8 estimated 15.94 GiB. Treat timings as a capacity/correctness record for
this machine, not a general speed ranking.

### Qwen2.5-7B per-prompt results

| Format | Prompt | Input tokens | Custom prefill / decode | cuBLAS prefill / decode | Identical tokens | Max abs logit diff | Non-finite (custom/cuBLAS) |
|---|---|---:|---:|---:|---:|---:|---|
| f32 | Tiny context | 33 | 283.67ms / 388.40ms | 84.50ms / 341.52ms | 9/9 | 8.774e-5 | 0 / 0 |
| f32 | Short context | 45 | 283.26ms / 648.19ms | 85.23ms / 569.78ms | 16/16 | 7.915e-5 | 0 / 0 |
| f32 | Medium context | 154 | 549.58ms / 653.69ms | 248.71ms / 575.58ms | 16/16 | 6.533e-5 | 0 / 0 |
| f32 | Long context | 321 | 1.05s / 662.03ms | 403.34ms / 584.24ms | 16/16 | 2.222e-4 | 0 / 0 |
| f32 | Long context with repeated structure | 282 | 958.67ms / 660.53ms | 368.16ms / 583.15ms | 16/16 | 1.450e-4 | 0 / 0 |
| f32 | Multi-step arithmetic | 56 | 285.58ms / 649.57ms | 86.50ms / 570.69ms | 16/16 | 1.278e-4 | 0 / 0 |
| f32 | Compare totals | 73 | 451.88ms / 649.73ms | 126.00ms / 571.74ms | 16/16 | 7.725e-5 | 0 / 0 |
| f32 | Exact format | 73 | 451.77ms / 649.66ms | 125.98ms / 571.27ms | 16/16 | 8.583e-5 | 0 / 0 |
| f32 | Time extraction | 78 | 452.38ms / 649.84ms | 126.41ms / 571.57ms | 16/16 | 1.161e-4 | 0 / 0 |
| fp16 | Tiny context | 33 | 310.06ms / 504.97ms | 38.13ms / 231.28ms | 9/9 | 1.005e-2 | 0 / 0 |
| fp16 | Short context | 45 | 325.46ms / 841.92ms | 39.32ms / 386.28ms | 16/16 | 9.233e-3 | 0 / 0 |
| fp16 | Medium context | 154 | 546.47ms / 848.64ms | 79.10ms / 390.81ms | 16/16 | 1.085e-2 | 0 / 0 |
| fp16 | Long context | 321 | 1.11s / 856.44ms | 105.63ms / 400.22ms | 16/16 | 8.895e-3 | 0 / 0 |
| fp16 | Long context with repeated structure | 282 | 969.26ms / 854.34ms | 100.64ms / 399.59ms | 16/16 | 8.425e-3 | 0 / 0 |
| fp16 | Multi-step arithmetic | 56 | 326.47ms / 843.24ms | 40.31ms / 386.98ms | 16/16 | 1.926e-2 | 0 / 0 |
| fp16 | Compare totals | 73 | 414.71ms / 843.97ms | 55.34ms / 388.13ms | 16/16 | 1.248e-2 | 0 / 0 |
| fp16 | Exact format | 73 | 414.90ms / 843.66ms | 59.39ms / 387.60ms | 16/16 | 1.847e-2 | 0 / 0 |
| fp16 | Time extraction | 78 | 415.52ms / 844.31ms | 55.89ms / 389.32ms | 16/16 | 1.138e-2 | 0 / 0 |
| int8 | Tiny context | 33 | 295.52ms / 210.67ms | 183.18ms / 1.12s | 9/9 | 8.011e-5 | 0 / 0 |
| int8 | Short context | 45 | 310.18ms / 362.59ms | 183.17ms / 1.87s | 16/16 | 8.249e-5 | 0 / 0 |
| int8 | Medium context | 154 | 533.08ms / 368.74ms | 348.35ms / 1.88s | 16/16 | 9.191e-5 | 0 / 0 |
| int8 | Long context | 321 | 1.12s / 375.35ms | 504.36ms / 1.89s | 16/16 | 8.726e-5 | 0 / 0 |
| int8 | Long context with repeated structure | 282 | 950.00ms / 375.18ms | 472.06ms / 1.89s | 16/16 | 2.232e-4 | 0 / 0 |
| int8 | Multi-step arithmetic | 56 | 311.24ms / 363.85ms | 187.05ms / 1.88s | 16/16 | 1.984e-4 | 0 / 0 |
| int8 | Compare totals | 73 | 407.24ms / 365.30ms | 233.06ms / 1.88s | 16/16 | 1.311e-4 | 0 / 0 |
| int8 | Exact format | 73 | 406.08ms / 365.11ms | 227.57ms / 1.87s | 16/16 | 1.450e-4 | 0 / 0 |
| int8 | Time extraction | 78 | 413.46ms / 363.61ms | 223.75ms / 1.88s | 16/16 | 8.440e-5 | 0 / 0 |

### Qwen2.5-14B per-prompt results

| Format | Prompt | Input tokens | Custom prefill / decode | cuBLAS prefill / decode | Identical tokens | Max abs logit diff | Non-finite (custom/cuBLAS) |
|---|---|---:|---:|---:|---:|---:|---|
| fp16 | Tiny context | 33 | 528.77ms / 1.47s | 79.61ms / 634.09ms | 13/13 | 1.012e-2 | 0 / 0 |
| fp16 | Short context | 45 | 540.05ms / 1.70s | 83.48ms / 731.39ms | 16/16 | 2.271e-2 | 0 / 0 |
| fp16 | Medium context | 154 | 1.17s / 1.71s | 154.17ms / 741.26ms | 16/16 | 1.431e-2 | 0 / 0 |
| fp16 | Long context | 321 | 2.15s / 1.72s | 225.92ms / 754.32ms | 16/16 | 8.244e-3 | 0 / 0 |
| fp16 | Long context with repeated structure | 282 | 2.02s / 1.72s | 206.82ms / 751.57ms | 16/16 | 1.356e-2 | 0 / 0 |
| fp16 | Multi-step arithmetic | 56 | 544.91ms / 1.70s | 85.03ms / 733.29ms | 16/16 | 1.783e-2 | 0 / 0 |
| fp16 | Compare totals | 73 | 843.51ms / 1.70s | 122.30ms / 733.10ms | 16/16 | 1.493e-2 | 0 / 0 |
| fp16 | Exact format | 73 | 843.24ms / 1.70s | 120.38ms / 733.99ms | 16/16 | 1.452e-2 | 0 / 0 |
| fp16 | Time extraction | 78 | 847.20ms / 1.59s | 121.57ms / 685.84ms | 14/14 | 1.714e-2 | 0 / 0 |
| int8 | Tiny context | 33 | 487.91ms / 581.02ms | 351.43ms / 3.16s | 13/13 | 8.345e-5 | 0 / 0 |
| int8 | Short context | 45 | 503.67ms / 688.93ms | 365.30ms / 3.65s | 16/16 | 8.011e-5 | 0 / 0 |
| int8 | Medium context | 154 | 1.10s / 697.73ms | 679.68ms / 3.66s | 16/16 | 1.297e-4 | 0 / 0 |
| int8 | Long context | 321 | 2.07s / 711.96ms | 946.16ms / 3.67s | 16/16 | 7.439e-5 | 0 / 0 |
| int8 | Long context with repeated structure | 282 | 1.93s / 708.51ms | 923.46ms / 3.66s | 16/16 | 1.202e-4 | 0 / 0 |
| int8 | Multi-step arithmetic | 56 | 509.44ms / 689.58ms | 352.42ms / 3.65s | 16/16 | 1.154e-4 | 0 / 0 |
| int8 | Compare totals | 73 | 763.48ms / 691.58ms | 420.70ms / 3.65s | 16/16 | 8.249e-5 | 0 / 0 |
| int8 | Exact format | 73 | 764.94ms / 691.55ms | 419.17ms / 3.65s | 16/16 | 1.137e-4 | 0 / 0 |
| int8 | Time extraction | 78 | 765.73ms / 647.28ms | 421.63ms / 3.41s | 14/14 | 1.488e-4 | 0 / 0 |

### Unit and model checks

| Check | Result |
|---|---|
| `cargo test --all-targets` | Passed: 2 unit tests (FP32→FP16 normal/subnormal/special encodings; independent rowwise INT8 scales and zero rows), 0 failed; each example target has 0 tests. Existing dead-code warnings remain. |
| `--check-model` on Qwen2.5-7B-Instruct | Passed: 28 layers, 339 tensors, 7,615,616,512 parameters; untied output head validated. |
| `--check-model` on Qwen2.5-14B-Instruct | Passed: 48 layers, 579 tensors, 14,770,033,664 parameters; untied output head validated. |
| Qwen2.5-1.5B and 3B | Full nine-prompt suites completed in F32, FP16, and INT8 on custom/cuBLAS; per-prompt evidence is in the CSV and raw logs. |
| Qwen2.5-7B | Full nine-prompt suites completed in F32, FP16, and INT8 on custom/cuBLAS. |
| Qwen2.5-14B | Full nine-prompt suites completed in FP16 and INT8 on custom/cuBLAS. |


## Same-precision cross-model comparison: FP16

**Precision/configuration audit:** all four checkpoint logs report `--weight-format fp16`, use GPU 1, `--max-tokens 16`, `--linear both`, and run all nine prompts. The saved commands match exactly apart from the model directory. No rerun was needed to make precision parity consistent. GPU load was not controlled between these single-pass runs, so timing differences are observations rather than isolated scaling claims. The original raw logs and every prompt row are preserved in [`fp16_comparison.csv`](benchmark-results/precision-2026-10-20/fp16_comparison.csv).

Reproduction for each model (replace the model directory):

```bash
cargo run --release -- --gpu 1 --model-dir models/Qwen2.5-1.5B-Instruct \
  --max-tokens 16 --linear both --weight-format fp16
```

### Per-prompt custom/cuBLAS timings

All times are milliseconds unless suffixed `s`. Each row reports custom prefill/decode, cuBLAS prefill/decode, generated-token agreement, and max absolute prefill-logit difference.

| Model | Prompt | Input tokens | Custom prefill / decode | cuBLAS prefill / decode | Matching tokens | Max abs logit diff |
|---|---|---:|---:|---:|---:|---:|
| 1.5B | Tiny context | 33 | 106.00ms / 231.86ms | 13.93ms / 118.96ms | 13/13 | 9.012e-3 |
| 1.5B | Short context | 45 | 107.96ms / 268.55ms | 13.55ms / 138.88ms | 16/16 | 8.146e-3 |
| 1.5B | Medium context | 154 | 154.46ms / 273.94ms | 24.21ms / 145.47ms | 16/16 | 1.386e-2 |
| 1.5B | Long context | 321 | 252.87ms / 282.47ms | 36.49ms / 151.59ms | 16/16 | 8.789e-3 |
| 1.5B | Long context with repeated structure | 282 | 224.67ms / 280.40ms | 34.16ms / 150.35ms | 16/16 | 9.294e-3 |
| 1.5B | Multi-step arithmetic | 56 | 108.67ms / 269.10ms | 14.10ms / 143.61ms | 16/16 | 7.067e-3 |
| 1.5B | Compare totals | 73 | 118.49ms / 269.34ms | 17.35ms / 145.70ms | 16/16 | 1.175e-2 |
| 1.5B | Exact format | 73 | 118.36ms / 269.42ms | 17.10ms / 162.49ms | 16/16 | 1.389e-2 |
| 1.5B | Time extraction | 78 | 118.71ms / 270.80ms | 17.47ms / 146.88ms | 16/16 | 7.511e-3 |
| 3B | Tiny context | 33 | 187.58ms / 335.33ms | 22.71ms / 159.66ms | 11/11 | 1.225e-2 |
| 3B | Short context | 45 | 192.05ms / 458.29ms | 22.51ms / 217.65ms | 16/16 | 1.793e-2 |
| 3B | Medium context | 154 | 254.89ms / 465.72ms | 39.88ms / 225.20ms | 16/16 | 1.170e-2 |
| 3B | Long context | 321 | 486.10ms / 476.64ms | 58.14ms / 236.05ms | 16/16 | 9.747e-3 |
| 3B | Long context with repeated structure | 282 | 409.35ms / 473.85ms | 55.68ms / 233.44ms | 16/16 | 1.373e-2 |
| 3B | Multi-step arithmetic | 56 | 193.63ms / 459.10ms | 23.13ms / 219.29ms | 16/16 | 1.568e-2 |
| 3B | Compare totals | 73 | 223.12ms / 460.20ms | 29.50ms / 220.19ms | 16/16 | 1.305e-2 |
| 3B | Exact format | 73 | 223.80ms / 458.84ms | 29.35ms / 219.51ms | 16/16 | 3.186e-2 |
| 3B | Time extraction | 78 | 223.16ms / 460.44ms | 29.65ms / 221.82ms | 16/16 | 1.268e-2 |
| 7B | Tiny context | 33 | 310.06ms / 504.97ms | 38.13ms / 231.28ms | 9/9 | 1.005e-2 |
| 7B | Short context | 45 | 325.46ms / 841.92ms | 39.32ms / 386.28ms | 16/16 | 9.233e-3 |
| 7B | Medium context | 154 | 546.47ms / 848.64ms | 79.10ms / 390.81ms | 16/16 | 1.085e-2 |
| 7B | Long context | 321 | 1.11s / 856.44ms | 105.63ms / 400.22ms | 16/16 | 8.895e-3 |
| 7B | Long context with repeated structure | 282 | 969.26ms / 854.34ms | 100.64ms / 399.59ms | 16/16 | 8.425e-3 |
| 7B | Multi-step arithmetic | 56 | 326.47ms / 843.24ms | 40.31ms / 386.98ms | 16/16 | 1.926e-2 |
| 7B | Compare totals | 73 | 414.71ms / 843.97ms | 55.34ms / 388.13ms | 16/16 | 1.248e-2 |
| 7B | Exact format | 73 | 414.90ms / 843.66ms | 59.39ms / 387.60ms | 16/16 | 1.847e-2 |
| 7B | Time extraction | 78 | 415.52ms / 844.31ms | 55.89ms / 389.32ms | 16/16 | 1.138e-2 |
| 14B | Tiny context | 33 | 528.77ms / 1.47s | 79.61ms / 634.09ms | 13/13 | 1.012e-2 |
| 14B | Short context | 45 | 540.05ms / 1.70s | 83.48ms / 731.39ms | 16/16 | 2.271e-2 |
| 14B | Medium context | 154 | 1.17s / 1.71s | 154.17ms / 741.26ms | 16/16 | 1.431e-2 |
| 14B | Long context | 321 | 2.15s / 1.72s | 225.92ms / 754.32ms | 16/16 | 8.244e-3 |
| 14B | Long context with repeated structure | 282 | 2.02s / 1.72s | 206.82ms / 751.57ms | 16/16 | 1.356e-2 |
| 14B | Multi-step arithmetic | 56 | 544.91ms / 1.70s | 85.03ms / 733.29ms | 16/16 | 1.783e-2 |
| 14B | Compare totals | 73 | 843.51ms / 1.70s | 122.30ms / 733.10ms | 16/16 | 1.493e-2 |
| 14B | Exact format | 73 | 843.24ms / 1.70s | 120.38ms / 733.99ms | 16/16 | 1.452e-2 |
| 14B | Time extraction | 78 | 847.20ms / 1.59s | 121.57ms / 685.84ms | 14/14 | 1.714e-2 |

All 36 model/prompt comparisons had identical custom/cuBLAS token IDs and zero non-finite logits. The full `.log` files also contain the generated text for both backends.

## Static independent-request GPU batches (2026-10-20)

Implementation commit: `09b2b4e` (`feat(inference): add static GPU request
batching`). Summary: add a static request-batch decode path and the
`--request-batch-sizes` CLI option. Each row is an independent prompt with its
own `KVCache`; its position is tracked separately, so prompts with different
lengths can decode in one request batch. The batch path runs projection GEMMs,
RMSNorm, MLP projections/SwiGLU, final logits, and argmax over request rows.
Custom and cuBLAS paths use the same batch protocol. Untied `lm_head` weights
are also used for the batched logits projection.

### Reproduction and measurement

All four commands used the same configuration except for the checkpoint:

```bash
cargo run --release -- --gpu 1 --model-dir models/Qwen2.5-1.5B-Instruct \
  --max-tokens 16 --linear both --weight-format fp16 \
  --request-batch-sizes 1,2,5,8
```

Environment: GPU 1, Tesla V100-SXM2-32GB, CUDA 12.4 / driver 550.144.03;
FP16 projection weights with FP32 activations and accumulation. The full
`BASE_PROMPTS.txt` suite ran first on both backends for every model, followed
by each static request batch. Batch sizes 1, 2, 5, and 8 each used a front and
tail prompt window (size 1 uses the first and last prompt). The size-5 and
size-8 windows contain short and long prompts. All requests are present before
prefill starts; no requests arrive during generation.

Prefill intentionally reuses the existing single-request prefill path and is
reported per request. During decode, projection, normalization, MLP, logits,
and argmax operate over the GPU request dimension. The current attention
implementation launches its CUDA attention kernel once per request because
each cache remains an independent allocation. The batch decode time includes
all decode model work, batched argmax, host token-ID transfer, and a stream
synchronization. Aggregate decode tok/s is total returned tokens divided by
that decode duration; serial initial prefill is excluded.

### Correctness and per-case evidence

| Model | Layers / hidden | Standalone prompt suite | Static batch groups | Batched requests | Batch sizes | GPU memory result |
|---|---:|---:|---:|---:|---|---|
| Qwen2.5-1.5B | 28 / 1536 | 9/9 prompts, custom/cuBLAS tokens identical | 16/16 passed | 64/64 identical to single-request reference | 1, 2, 5, 8 | Completed |
| Qwen2.5-3B | 36 / 2048 | 9/9 prompts, custom/cuBLAS tokens identical | 16/16 passed | 64/64 identical to single-request reference | 1, 2, 5, 8 | Completed |
| Qwen2.5-7B | 28 / 3584 | 9/9 prompts, custom/cuBLAS tokens identical | 16/16 passed | 64/64 identical to single-request reference | 1, 2, 5, 8 | Completed |
| Qwen2.5-14B | 48 / 5120 | 9/9 prompts, custom/cuBLAS tokens identical | 16/16 passed | 64/64 identical to single-request reference | 1, 2, 5, 8 | Batch 8 completed; no OOM |

Each model ran 8 cases per backend: two prompt windows at each of four batch
sizes. Every request's generated token IDs exactly matched a fresh single-
request run on the same backend; the program exits with an error on a
mismatch. Every standalone custom/cuBLAS prompt comparison was token-identical
with zero non-finite logits. The prompt-level single-run timings are in
[`full_prompt_results.csv`](benchmark-results/request-batching-2026-10-20/full_prompt_results.csv).
The complete, machine-readable group timings and request-level timings are in
[`batch_case_results.csv`](benchmark-results/request-batching-2026-10-20/batch_case_results.csv)
and
[`per_request_results.csv`](benchmark-results/request-batching-2026-10-20/per_request_results.csv).
The four `.log` files alongside those CSVs retain command output, generated
text, correctness checks, and all nine prompt timings for each model.

### Decode batch throughput observations

The table gives the two measured front/tail windows as a range. Values are
aggregate returned output tokens per second; exact group timings and every
prompt membership are in `batch_case_results.csv`.

| Model | Backend | Batch 1 | Batch 2 | Batch 5 | Batch 8 |
|---|---|---:|---:|---:|---:|
| 1.5B | custom | 54.2–57.5 | 19.2–21.2 | 48.4–50.9 | 75.7–77.1 |
| 1.5B | cuBLAS | 98.2–102.5 | 155.4–173.5 | 297.2–321.2 | 432.4–433.5 |
| 3B | custom | 32.0–34.0 | 10.3–12.2 | 27.5–29.6 | 43.6–45.3 |
| 3B | cuBLAS | 65.3–68.5 | 98.1–115.0 | 208.4–232.3 | 296.1–307.2 |
| 7B | custom | 17.6–18.7 | 7.0–8.9 | 19.8–21.9 | 31.9–33.8 |
| 7B | cuBLAS | 37.9–40.3 | 57.5–73.2 | 141.9–160.4 | 216.6–227.9 |
| 14B | custom | 8.7 | 4.1–4.2 | 10.7–10.9 | 16.9–17.0 |
| 14B | cuBLAS | 19.9–20.0 | 34.0–34.9 | 78.2–81.0 | 117.0–117.3 |

These are single-pass observations on a shared host. The custom GEMM path is
especially inefficient at small request counts, and attention remains
request-by-request; the numbers show current behavior rather than an isolated
kernel speedup. Larger request groups improve cuBLAS aggregate decode
throughput on these cases. Initial prefill is serial, so end-to-end request
batch latency includes the sum of per-prompt prefill times shown per request
in the CSV.

### Validation

| Check | Result |
|---|---|
| `cargo test --all-targets` | Passed: 2 unit tests; binary and example targets passed with 0 tests. |
| `cargo check` | Passed. |
| Standalone prompt suite, 1.5B/3B/7B/14B | 9 prompts per model on both backends; custom/cuBLAS outputs identical, zero non-finite logits. |
| Static request batches | 64 cases, 256 request records; every generated sequence matched its independent single-request reference. |
| Batch size 5 and mixed short/long prompts | Passed on both backends for every available model. |
| Batch size 8 | Passed on both backends for every available model, including 14B FP16. |
