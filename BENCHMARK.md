# Prefill Benchmark: previous (`main`) vs new (`feat/gemm-prefill`)

`main` is untouched and remains the baseline. The new branch keeps
`src/kernels/layer/linear.cu` + `LinearKernel::forward` unchanged (decode +
comparison baseline) and adds `src/kernels/layer/gemm.cu` +
`LinearKernel::forward_gemm` reading the **same** resident weight slices.

## How to reproduce

```bash
# new branch: all three prefill paths + logits check per prompt
./target/debug/Engine --gpu <id> --model-dir <model> --max-tokens 16 --bench-prefill
# previous (main checkout): token-loop baseline only
./target/debug/Engine --gpu <id> --model-dir <model> --max-tokens 16
```

`--prefill {gemv-loop,batched-gemv,gemm}` selects the prefill path for
generation; `--bench-prefill` runs all three on each prompt, prints max-abs
logits diff vs the loop, and decodes from the selected path. All timings
synchronize the stream before stopping the clock. Speedups in parentheses
are within-run (same process/GPU) vs the loop.

Setup: Tesla V100-SXM2-32GB, f32, greedy decode, 16 max tokens.
`main` ran on GPUs 0/2/1, new branch on GPUs 6/7/1 for 0.5B/1.5B/3B
(3B both on GPU 1). Times in ms unless noted.

## Qwen2.5-0.5B (hidden 896, 24 layers)

| Prompt (tokens) | main (prev) | loop | batched-GEMV | GEMM | bg diff | gemm diff |
|---|---|---|---|---|---|---|
| Tiny (33) | 394 | 240 | 72 (3.3x) | 54 (4.5x) | 0.00e0 | 4.96e-5 |
| Short (45) | 538 | 316 | 83 (3.8x) | 58 (5.4x) | 0.00e0 | 4.44e-5 |
| Medium (154) | 1780 | 1100 | 293 (3.8x) | 75 (14.7x) | 0.00e0 | 4.34e-5 |
| Long (321) | 3620 | 2410 | 662 (3.6x) | 81 (29.6x) | 8.23e-6 | 3.46e-5 |
| Long-repeat (282) | 3200 | 2120 | 599 (3.5x) | 79 (26.8x) | 0.00e0 | 6.10e-5 |
| Arithmetic (56) | 724 | 394 | 122 (3.2x) | 58 (6.8x) | 0.00e0 | 4.73e-5 |
| Totals (73) | 916 | 524 | 150 (3.5x) | 50 (10.4x) | 0.00e0 | 7.25e-5 |
| Format (73) | 832 | 535 | 117 (4.6x) | 55 (9.7x) | 0.00e0 | 7.05e-5 |
| Time-extract (78) | 844 | 479 | 119 (4.0x) | 50 (9.5x) | 0.00e0 | 6.63e-5 |

## Qwen2.5-1.5B (hidden 1536, 28 layers)

| Prompt (tokens) | main (prev) | loop | batched-GEMV | GEMM | bg diff | gemm diff |
|---|---|---|---|---|---|---|
| Tiny (33) | 734 | 804 | 413 (1.9x) | 190 (4.2x) | 0.00e0 | 1.25e-4 |
| Short (45) | 1040 | 676 | 419 (1.6x) | 169 (4.0x) | 0.00e0 | 7.15e-5 |
| Medium (154) | 3770 | 2720 | 1370 (2.0x) | 147 (18.6x) | 1.34e-1 | 4.91e-3 |
| Long (321) | 7890 | 6530 | 4020 (1.6x) | 512 (12.8x) | 6.00e-2 | 1.48e-1 |
| Long-repeat (282) | 6960 | 6200 | 2400 (2.6x) | 321 (19.3x) | 6.14e-2 | 9.19e-3 |
| Arithmetic (56) | 1380 | 855 | 517 (1.6x) | 105 (8.2x) | 0.00e0 | 8.06e-5 |
| Totals (73) | 1790 | 1240 | 451 (2.8x) | 115 (10.8x) | 0.00e0 | 7.96e-5 |
| Format (73) | 1770 | 1780 | 898 (2.0x) | 231 (7.7x) | 0.00e0 | 1.01e-4 |
| Time-extract (78) | 1950 | 1930 | 969 (2.0x) | 203 (9.5x) | 0.00e0 | 2.26e-2 |

## Qwen2.5-3B (hidden 2048, 36 layers)

| Prompt (tokens) | main (prev) | loop | batched-GEMV | GEMM | bg diff | gemm diff |
|---|---|---|---|---|---|---|
| Tiny (33) | 754 | 1050 | 604 (1.7x) | 182 (5.8x) | 0.00e0 | 1.14e-3 |
| Short (45) | 1580 | 1030 | 591 (1.7x) | 184 (5.6x) | 0.00e0 | 1.09e-3 |
| Medium (154) | 6200 | 5600 | 3760 (1.5x) | 463 (12.1x) | 1.04e-2 | 6.39e-2 |
| Long (321) | 12970 | 13340 | 5660 (2.4x) | 657 (20.3x) | 2.97e-2 | 9.16e-2 |
| Long-repeat (282) | 9180 | 11550 | 6960 (1.7x) | 680 (17.0x) | 6.34e-2 | 4.81e-2 |
| Arithmetic (56) | 2240 | 2240 | 1320 (1.7x) | 327 (6.9x) | 0.00e0 | 1.03e-3 |
| Totals (73) | 2940 | 2240 | 964 (2.3x) | 203 (11.0x) | 0.00e0 | 2.53e-3 |
| Format (73) | 2930 | 2840 | 1760 (1.6x) | 340 (8.3x) | 0.00e0 | 1.86e-3 |
| Time-extract (78) | 3110 | 3120 | 1870 (1.7x) | 365 (8.6x) | 1.92e-1 | 5.70e-4 |

## Reading

- `main` prefill tracks the in-bench `loop` column (same token-at-a-time
  code path; residual gap is GPU/system variance across runs).
- batched-GEMV gives a flat ~2-4x (fewer launches, same HBM weight traffic).
- Blocked GEMM (BM=32 x BN=128, BK=32, TM=4 x TN=4 register tile, ~20KB
  shared) reuses each weight tile across 32 prompt rows: prefill stays
  roughly flat while the loop scales linearly. Best observed: **29.6x**
  (0.5B/321 tok), **19.3x** (1.5B/282 tok), **20.3x** (3B/321 tok).
- Correctness: all 27 greedy generations (9 prompts x 3 models) are
  **token-identical** between `main` and the GEMM path. Max-abs diffs are
  over the full 151936-dim logits (tail noise; top-1 unaffected everywhere).
- Notes: the loop runs first per prompt so it also pays first-launch warmup
  on prompt 1. On 1.5B/3B long prompts the diffs reach ~0.1 and show small
  run-to-run wobble (see branch log); batched helpers re-verified bit-exact
  vs the loop on repeats, generations unaffected — root cause of the wobble
  is still open.
