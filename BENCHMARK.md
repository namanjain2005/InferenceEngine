# Prefill Benchmark: GEMV-loop (baseline) vs batched-GEMV vs blocked GEMM

Branch: `feat/gemm-prefill` — `main` is untouched and remains the baseline.
`src/kernels/layer/linear.cu` and `LinearKernel::forward` are unchanged;
the new path adds `src/kernels/layer/gemm.cu` + `LinearKernel::forward_gemm`
reading the **same** resident weight slices. Decode always uses GEMV.

## How to reproduce

```bash
cargo build
./target/debug/Engine --gpu 1 \
  --model-dir /home/23uec572/Naman_jain/InferenceEngine/models/Qwen2.5-0.5B-Instruct \
  --max-tokens 16 --bench-prefill
```

`--prefill {gemv-loop,batched-gemv,gemm}` selects the prefill path for
generation; `--bench-prefill` runs all three on each prompt, prints max-abs
logits diff vs the loop baseline, and decodes from the selected path.
All timings synchronize the stream before stopping the clock.

## Results (Tesla V100-SXM2-32GB, Qwen2.5-0.5B, f32)

| Prompt tokens | gemv-loop | batched-gemv | speedup | max diff | gemm | speedup | max diff |
|---|---|---|---|---|---|---|---|
| 33 | 361.74 ms | 66.94 ms | 5.40x | 0.00e0 | 88.56 ms | 4.08x | 4.96e-5 |
| 45 | 471.54 ms | 115.30 ms | 4.09x | 0.00e0 | 86.81 ms | 5.43x | 4.44e-5 |
| 154 | 1.65 s | 457.33 ms | 3.60x | 0.00e0 | 79.37 ms | 20.74x | 4.34e-5 |
| 321 | 3.53 s | 958.18 ms | 3.68x | 8.54e-4 | 137.07 ms | 25.75x | 3.24e-5 |
| 282 | 3.04 s | 817.38 ms | 3.72x | 4.29e-4 | 146.22 ms | 20.80x | 2.15e-4 |
| 56 | 584.92 ms | 137.31 ms | 4.26x | 0.00e0 | 83.82 ms | 6.98x | 4.73e-5 |

## Reading

- batched-GEMV gives a flat ~4x (fewer launches, same HBM weight traffic).
- Blocked GEMM (BM=32 x BN=128, BK=32, TM=4 x TN=4 register tile, ~20KB
  shared) reuses each weight tile across 32 prompt rows, so prefill time
  stays roughly constant while the loop scales linearly: **~21-26x** on
  150-320 token prompts.
- Logits diffs vs baseline are float reassociation noise (reduction order);
  batched-GEMV is bit-exact on short prompts.
- Note: the loop runs first on prompt 1, so it also pays first-launch
  warmup there; the long-prompt speedups are well beyond warmup noise.
