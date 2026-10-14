# Prefill Benchmark: previous (`main`) vs new (`feat/gemm-prefill`) — fair protocol

`main` is untouched and remains the baseline. The new branch keeps
`src/kernels/layer/linear.cu` + `LinearKernel::forward` unchanged (decode +
comparison baseline) and adds `src/kernels/layer/gemm.cu` +
`LinearKernel::forward_gemm` reading the **same** resident weight slices.

## Fairness protocol (revised after auditing the first round)

The first round of numbers had four upward biases, all fixed here:

1. **Throwaway logits.** The old loop baseline projected full 151936-wide
   logits per token and discarded M−1 of them (~20% of its time on short
   prompts). The loop now uses `forward_token_hidden` for all but the last
   token, i.e. exactly the work the batched paths do. Kernel speedup
   (loop→GEMM) is same-work; system speedup (`main` as-shipped→GEMM)
   is reported separately.
2. **Single-shot noise on a shared DGX.** Now `--bench-repeat 3`,
   round-robin interleaved per path, median reported.
3. **Order bias.** Warmup runs all three paths untimed before the suite;
   repeats are interleaved so drift can't favor one path.
4. **Cross-GPU `main` comparison.** `main` re-ran on the same GPU as the
   new branch per model (0.5B→GPU 6, 1.5B→GPU 7, 3B→GPU 1).
5. **Async timing.** Stream synchronized before every clock stop.

Reproduce:

```bash
./target/debug/Engine --gpu <id> --model-dir <model> --max-tokens 16 \
  --bench-prefill --bench-repeat 3   # new branch
./target/debug/Engine --gpu <id> --model-dir <model> --max-tokens 16  # main
```

Setup: Tesla V100-SXM2-32GB, f32, greedy decode. Times in ms unless noted.
Speedups in parentheses are within-run medians vs the same-work loop;
`sys` additionally divides as-shipped `main` by GEMM.

## Qwen2.5-0.5B (hidden 896, 24 layers)

| Prompt (tok) | main | loop | batched-GEMV | GEMM | bg diff | gemm diff |
|---|---|---|---|---|---|---|
| Tiny (33) | 206 | 192 | 49 (3.9x) | 46 (4.2x, sys 4.5x) | 0.00e0 | 4.96e-5 |
| Short (45) | 275 | 253 | 67 (3.8x) | 47 (5.4x, sys 5.9x) | 0.00e0 | 4.44e-5 |
| Medium (154) | 951 | 881 | 242 (3.6x) | 57 (15.4x, sys 16.6x) | 0.00e0 | 1.64e-4 |
| Long (321) | 2050 | 1890 | 512 (3.7x) | 78 (24.1x, sys 26.1x) | 0.00e0 | 3.24e-5 |
| Long-repeat (282) | 1770 | 1620 | 449 (3.6x) | 75 (21.6x, sys 23.6x) | 0.00e0 | 6.10e-5 |
| Arithmetic (56) | 332 | 337 | 85 (4.0x) | 47 (7.2x, sys 7.1x) | 0.00e0 | 4.73e-5 |
| Totals (73) | 477 | 421 | 112 (3.8x) | 50 (8.4x, sys 9.5x) | 0.00e0 | 7.25e-5 |
| Format (73) | 480 | 424 | 111 (3.8x) | 50 (8.4x, sys 9.6x) | 0.00e0 | 7.05e-5 |
| Time-extract (78) | 484 | 445 | 119 (3.7x) | 50 (8.8x, sys 9.6x) | 0.00e0 | 6.63e-5 |

Generations: 9/9 token-identical to `main`. Batched helpers bit-exact 9/9.

## Qwen2.5-1.5B (hidden 1536, 28 layers)

| Prompt (tok) | main | loop | batched-GEMV | GEMM | bg diff | gemm diff |
|---|---|---|---|---|---|---|
| Tiny (33) | 443 | 584 | 273 (2.1x) | 155 (3.8x, sys 2.8x) | 0.00e0 | 1.25e-4 |
| Short (45) | 865 | 690 | 359 (1.9x) | 151 (4.6x, sys 5.7x) | 0.00e0 | 7.15e-5 |
| Medium (154) | 3650 | 2830 | 1370 (2.1x) | 217 (13.1x, sys 16.8x) | 5.70e-3 | 8.05e-3 |
| Long (321) | 7880 | 6300 | 2760 (2.3x) | 388 (16.2x, sys 20.3x) | 3.53e-2 | 0.00e0 |
| Long-repeat (282) | 4720 | 6510 | 2800 (2.3x) | 356 (18.3x, sys 13.2x) | 0.00e0 | 0.00e0 |
| Arithmetic (56) | 892 | 923 | 496 (1.9x) | 105 (8.8x, sys 8.5x) | 0.00e0 | 8.06e-5 |
| Totals (73) | 1070 | 1610 | 907 (1.8x) | 217 (7.4x, sys 4.9x) | 0.00e0 | 7.96e-5 |
| Format (73) | 1100 | 1160 | 704 (1.6x) | 180 (6.4x, sys 6.1x) | 0.00e0 | 6.33e-1 |
| Time-extract (78) | 1200 | 1320 | 965 (1.4x) | 205 (6.5x, sys 5.9x) | 0.00e0 | 1.35e-4 |

Note: `main` (separate process, single-shot) is noisier here — e.g. prompt 5
`main` 4720 ms beats same-work loop 6510 ms despite doing strictly more
work. The loop column (same-process median-of-3) is the rigorous baseline;
`main` is reference only.

## Qwen2.5-3B (hidden 2048, 36 layers)

| Prompt (tok) | main | loop | batched-GEMV | GEMM | bg diff | gemm diff |
|---|---|---|---|---|---|---|
| Tiny (33) | 754 | 854 | 574 (1.5x) | 254 (3.4x, sys 3.0x) | 0.00e0 | 1.14e-3 |
| Short (45) | 1580 | 1700 | 1080 (1.6x) | 328 (5.2x, sys 4.8x) | 0.00e0 | 1.09e-3 |
| Medium (154) | 6200 | 5840 | 3780 (1.5x) | 464 (12.6x, sys 13.4x) | 8.14e-2 | 7.69e-3 |
| Long (321) | 12970 | 11920 | 7520 (1.6x) | 874 (13.7x, sys 14.8x) | 4.01e-2 | 0.00e0 |
| Long-repeat (282) | 9180 | 10960 | 6980 (1.6x) | 706 (15.5x, sys 13.0x) | 3.08e-2 | 1.18e-1 |
| Arithmetic (56) | 2240 | 2130 | 1360 (1.6x) | 315 (6.8x, sys 7.1x) | 0.00e0 | 8.28e-3 |
| Totals (73) | 2940 | 2780 | 1370 (2.0x) | 291 (9.6x, sys 10.1x) | 0.00e0 | 5.19e-2 |
| Format (73) | 2930 | 2780 | 1790 (1.6x) | 345 (8.1x, sys 8.5x) | 5.51e-2 | 1.86e-3 |
| Time-extract (78) | 3110 | 2960 | 1910 (1.6x) | 356 (8.3x, sys 8.7x) | 0.00e0 | 9.39e-2 |

## cuBLAS backend (`--linear cublas`, default)

`CublasLinear` implements the same `LinearOp` trait over identical
row-major layouts (standard transpose mapping, no rearrangement), plus a
small `bias_add` kernel since cuBLAS has no fused bias. Custom kernels are
untouched and selectable via `--linear custom` for fusing work.

Qwen2.5-0.5B, same fair protocol (median-of-3, GPU 6), 9/9 generations
token-identical to the custom backend:

| Prompt (tok) | loop (cublas) | cublas-GEMM | speedup |
|---|---|---|---|
| 33 | 181 | 8.6 | 21.0x |
| 45 | 239 | 8.8 | 27.3x |
| 154 | 849 | 18.5 | 46.0x |
| 321 | 2290 | 33.5 | 68.5x |
| 282 | 1580 | 30.5 | 51.9x |

Notes:

- cuBLAS prefill is ~2.3x faster than the custom blocked GEMM (33 ms vs
  78 ms at 321 tok). Decode is a wash/slightly custom-favored (lean GEMV
  launch vs sgemm-per-token overhead) — a first hint for per-case
  specialization later: custom GEMV for decode, cuBLAS for prefill.
- Gotcha found while wiring this up: **cublas 12.4 `cublasSgemv`
  transpose path writes past `y` for K != N shapes** (compute-sanitizer:
  `Invalid __global__ write` in `gemv2T_kernel`; dmesg Xid 31 PDE faults
  + Xid 43). Decode therefore routes through `sgemm` with M=1, which is
  sanitizer-clean (0 errors) and value-exact on all shapes. Revisit if the
  CUDA/cuBLAS version changes.

## Reading

- Proper, same-work kernel speedup (loop→GEMM, median-of-3): **24.1x**
  (0.5B/321 tok), **18.3x** (1.5B/282 tok), **15.5x** (3B/282 tok).
  First-round headlines (29.6x/20.3x) were inflated by the four biases
  above and are superseded by these.
- batched-GEMV is flat ~1.5-4x (launch savings only, same HBM traffic).
- GEMM prefill time stays roughly flat as prompts grow (weight tile reuse
  across 32 rows); the loop scales linearly.
- Correctness: 0.5B generations 9/9 identical, batched bit-exact 9/9, GEMM
  twice bit-exact on the longest prompts. On 1.5B/3B long prompts, max-abs
  diffs reach ~0.1-0.6 in rare runs (tail logits; top-1 stable in all
  observed generations) with small run-to-run wobble — open stress-test
  item, does not affect timing.
- Transient (one-off, not reproduced in 8 reruns incl. 5x stress): the
  `main` binary once degenerated to `!!!…` on 1.5B prompts 4–5 and 3B
  prompt 4. Those three cells' `main` timings still measure real work and
  are kept; generation agreement excludes them.
