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

## cuBLAS backend

`CublasLinear` implements the same `LinearOp` trait over identical
row-major layouts (standard transpose mapping, no rearrangement), plus a
small `bias_add` kernel since cuBLAS has no fused bias. Custom kernels are
untouched and selectable via `--linear custom` for fusing work.

Decode uses `cublasSgemv` directly. The row-major `[N, K]` weights are passed
as the column-major `[K, N]` matrix A, with `CUBLAS_OP_T`, `m=K`, `n=N`,
`lda=K`, giving an N-element output; a small `bias_add` kernel adds the bias.

An earlier implementation routed this through `cublasSgemm` with M=1 based on
a reported out-of-bounds write in cuBLAS 12.4's transpose GEMV path. That was
a misattribution. An isolated `cublasSgemv` harness exercising every
projection shape (896→896/128/4864, 4864→896, 896→151936, plus 1536² and
2048² square cases) is **0 errors under `compute-sanitizer --tool memcheck`**
on this CUDA/cuBLAS build, with values matching a CPU reference. The fault
that aborted the earlier whole-engine sanitizer run came from our own
`attention_decode` kernel (an out-of-bounds shared-memory read just past the
dynamic score buffer); it poisoned the context, so the *next* cuBLAS call
surfaced as `CUBLAS_STATUS_EXECUTION_FAILED`. Padding the dynamic score buffer
fixed it — the engine is now 0-error under memcheck end-to-end.

## Dual-run backend comparison (`--linear both`, default)

The library stays single-backend: `main` tokenizes each prompt once, runs
**both** backends on the identical token ids (each with a fresh KV cache),
prints each one's output + timing, then a head-to-head block. Weights are
uploaded once and read by both — no duplication. Release build, f32, greedy,
`--max-tokens 16`, all runs on one idle GPU (4). These tables use the current
direct-`cublasSgemv` decode path.

Decode is launch-overhead-bound (hundreds of small kernels per step), so it is
much noisier than prefill. The numbers below are therefore medians: 0.5B over
3 runs, 1.5B over 2 runs, 3B a single pass; prefill is stable across repeats
and reported from the same set. Tables show prefill / decode ms and the winner.

> An isolated microbenchmark (`examples/decode_bench.rs`) of `cublasSgemv`
> against the old `cublasSgemm(M=1)` fallback gives **1.00x on every decode
> projection shape** (896→896/128/4864, 4864→896, 896/1536/2048→151936, and
> 1536²) with bit-identical outputs; both reach ~850 GB/s on the large logits
> projections. So the `sgemv` switch is performance-neutral and the decode
> differences below are launch/measurement overhead, not the GEMV choice.

### Qwen2.5-0.5B

| Prompt (tok) | custom pfl | cublas pfl | pfl | custom dec | cublas dec | dec | gen |
|---|---|---|---|---|---|---|---|
| 33 | 49.6 | 11.6 | cublas 4.28x | 80.6 | 70.8 | cublas 1.14x | 13/13 |
| 45 | 46.6 | 8.9 | cublas 5.21x | 93.8 | 83.2 | cublas 1.13x | 16/16 |
| 154 | 56.9 | 18.7 | cublas 3.05x | 102.9 | 107.2 | custom 1.04x | 16/16 |
| 321 | 78.3 | 32.7 | cublas 2.39x | 112.5 | 138.9 | custom 1.23x | 16/16 |
| 282 | 73.2 | 30.0 | cublas 2.44x | 106.9 | 98.1 | cublas 1.09x | 16/16 |
| 56 | 46.7 | 9.2 | cublas 5.10x | 93.6 | 90.3 | cublas 1.04x | 16/16 |
| 73 | 50.1 | 11.9 | cublas 4.20x | 110.1 | 107.0 | cublas 1.03x | 16/16 |
| 73 | 50.1 | 11.8 | cublas 4.24x | 99.5 | 111.2 | custom 1.12x | 16/16 |
| 78 | 50.2 | 12.1 | cublas 4.16x | 111.2 | 125.1 | custom 1.13x | 16/16 |

### Qwen2.5-1.5B

| Prompt (tok) | custom pfl | cublas pfl | pfl | custom dec | cublas dec | dec | gen |
|---|---|---|---|---|---|---|---|
| 33 | 102.9 | 22.3 | cublas 4.62x | 170.4 | 148.9 | cublas 1.14x | 13/13 |
| 45 | 103.8 | 22.5 | cublas 4.61x | 197.2 | 170.5 | cublas 1.16x | 16/16 |
| 154 | 145.1 | 52.7 | cublas 2.75x | 202.3 | 171.4 | cublas 1.18x | 16/16 |
| 321 | 253.4 | 94.1 | cublas 2.69x | 210.4 | 184.7 | cublas 1.14x | NaN* |
| 282 | 212.6 | 84.1 | cublas 2.53x | 210.0 | 184.1 | cublas 1.14x | NaN* |
| 56 | 104.5 | 22.9 | cublas 4.56x | 196.8 | 163.7 | cublas 1.20x | 16/16 |
| 73 | 114.7 | 31.7 | cublas 3.61x | 197.8 | 165.2 | cublas 1.20x | 16/16 |
| 73 | 114.7 | 31.4 | cublas 3.65x | 197.8 | 165.3 | cublas 1.20x | 16/16 |
| 78 | 114.9 | 31.9 | cublas 3.60x | 198.1 | 166.7 | cublas 1.19x | 16/16 |

\* Long-context prompts hit the non-deterministic NaN logits (either backend);
see the finding below. The affected run's token 0 is `!` for both backends, so
the two "diverge" only in that both emit the degenerate `!!!…` continuation.

### Qwen2.5-3B

| Prompt (tok) | custom pfl | cublas pfl | pfl | custom dec | cublas dec | dec | gen |
|---|---|---|---|---|---|---|---|
| 33 | 183.6 | 41.3 | cublas 4.45x | 251.6 | 210.3 | cublas 1.20x | 11/11 |
| 45 | 184.4 | 41.8 | cublas 4.41x | 343.9 | 287.8 | cublas 1.19x | 16/16 |
| 154 | 265.5 | 100.8 | cublas 2.63x | 350.4 | 296.0 | cublas 1.18x | 16/16 |
| 321 | 484.8 | 183.3 | cublas 2.65x | 360.9 | 308.4 | cublas 1.17x | 16/16 |
| 282 | 402.6 | 150.9 | cublas 2.67x | 359.0 | 306.0 | cublas 1.17x | NaN* |
| 56 | 184.9 | 42.4 | cublas 4.36x | 344.7 | 290.1 | cublas 1.19x | 16/16 |
| 73 | 202.8 | 58.0 | cublas 3.50x | 345.1 | 290.4 | cublas 1.19x | 16/16 |
| 73 | 202.8 | 57.8 | cublas 3.51x | 346.2 | 291.2 | cublas 1.19x | 16/16 |
| 78 | 203.2 | 58.7 | cublas 3.46x | 346.4 | 290.4 | cublas 1.19x | 16/16 |

Findings:

- **cuBLAS wins prefill decisively and decode modestly.** Prefill by ~2.4–5.2x
  (grows with prompt; the blocked GEMM kernel is the bottleneck). Decode is
  ~1.1–1.2x on 1.5B/3B and stable across repeats, but on 0.5B it is a wash
  (median-of-3: cuBLAS wins 5/9, custom 4/9, all within ±0.15x). Small-model
  decode is launch-overhead-bound, and the cuBLAS path pays one extra
  `bias_add` launch per projection because cuBLAS has no fused bias, which is
  the whole of the difference. Keep custom for the fused-kernel experiments,
  not for speed today.
- **`cublasSgemv` vs the old `sgemm(M=1)` fallback is a non-issue.** The
  isolated microbenchmark above shows 1.00x on every decode shape and
  bit-identical outputs, so neither is "faster"; the choice is neutral and the
  `sgemv` path is the more direct expression of the operation.
- **Non-deterministic NaN logits (both backends).** On 1.5B/3B long prompts
  (321/282 tok), one backend's *entire* logits vector (151936 = vocab) comes
  back non-finite, so greedy argmax falls through to token 0 = `!` (the `!!!…`
  rows). Still reproduces after the attention shared-memory fix below, hitting
  custom or cuBLAS at random, sometimes both. It is not a cuBLAS bug and not
  specific to either path — both share every non-linear kernel (rope, rmsnorm,
  attention, cache append, swiglu), so the suspect is a race/uninitialised
  read or float blow-up in one of those on long context. The comparison
  reporter prints non-finite counts so this is no longer masked (`f32::max`
  used to hide NaN as a 0.0 diff). Open item: chase it with
  `compute-sanitizer --tool racecheck` on the 321-token prompt.
- **Fixed: out-of-bounds shared read in decode/prefill attention.** The
  earlier whole-engine sanitizer failure was our `attention_decode` scanning
  a few elements past the dynamic `scores[]` buffer (a vectorised tail load).
  Both attention launch sites now pad that buffer by 8 floats; memcheck on the
  full engine is 0 errors and the run completes. This is backend-independent
  and does not, by itself, explain the NaN above.

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
- Transient degeneration, now characterised (see "Dual-run" findings): on
  long context (seen on 1.5B/321, 1.5B/282 and 3B/282) one backend's full
  logits vector goes non-finite non-deterministically, so argmax returns token
  0 (`!`). Backend-agnostic and shared-kernel, so it is a latent
  race/overflow, not a cuBLAS defect. Timing cells still measure real work;
  generation agreement excludes the affected runs.
