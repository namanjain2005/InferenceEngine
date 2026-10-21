# Long-context FlashAttention comparison

Each timing is previous attention → FlashAttention, in milliseconds. Percentages compare the new value with the previous one.

The previous implementation is commit `47960ee`; FlashAttention is commit `d381615`. Both used the same expanded `BASE_PROMPTS.txt`, `--max-tokens 16`, `--linear both`, FP16 projection weights, and the same GPU for each model pair. GPU 3 ran 1.5B/3B/7B; GPU 4 ran 14B. The host was shared and GPU 3 was frequently busy, so these are single-pass measurements and small differences are noisy.

Input token counts were 1,006, 4,036, and 8,076 for the approximately 1K, 4K, and 8K cases. All 1.5B, 3B, and 7B cases completed on both revisions. Both 14B runs completed 1K, then failed at 4K during custom prefill with CUDA out of memory while allocating a GEMM output. The 14B 8K case was not reached. This is reported as a model working-memory limit; it is not a shared-memory launch failure from the attention kernel.

| Model | Context | Input tokens | Backend | Prefill old → new | Decode old → new | Output text | Status |
|---|---|---:|---|---:|---:|---|---|
| 1.5B | Approximately 1K-token context | 1006 | custom | 1310.00 → 1320.00 ms (+0.8%) | 471.34 → 468.94 ms (-0.5%) | IDENTICAL | matched |
| 1.5B | Approximately 1K-token context | 1006 | cublas | 244.43 → 263.25 ms (+7.7%) | 267.31 → 268.41 ms (+0.4%) | IDENTICAL | matched |
| 1.5B | Approximately 4K-token context | 4036 | custom | 6410.00 → 6480.00 ms (+1.1%) | 114.12 → 119.70 ms (+4.9%) | IDENTICAL | matched |
| 1.5B | Approximately 4K-token context | 4036 | cublas | 2480.00 → 2580.00 ms (+4.0%) | 82.35 → 80.46 ms (-2.3%) | IDENTICAL | matched |
| 1.5B | Approximately 8K-token context | 8076 | custom | 20620.00 → 17540.00 ms (-14.9%) | 165.62 → 168.39 ms (+1.7%) | IDENTICAL | matched |
| 1.5B | Approximately 8K-token context | 8076 | cublas | 13120.00 → 10020.00 ms (-23.6%) | 133.82 → 139.45 ms (+4.2%) | IDENTICAL | matched |
| 3B | Approximately 1K-token context | 1006 | custom | 2660.00 → 2680.00 ms (+0.8%) | 63.46 → 66.96 ms (+5.5%) | IDENTICAL | matched |
| 3B | Approximately 1K-token context | 1006 | cublas | 433.39 → 447.61 ms (+3.3%) | 37.94 → 35.79 ms (-5.7%) | IDENTICAL | matched |
| 3B | Approximately 4K-token context | 4036 | custom | 12470.00 → 12550.00 ms (+0.6%) | 86.14 → 91.52 ms (+6.2%) | IDENTICAL | matched |
| 3B | Approximately 4K-token context | 4036 | cublas | 4400.00 → 4480.00 ms (+1.8%) | 60.89 → 59.22 ms (-2.7%) | IDENTICAL | matched |
| 3B | Approximately 8K-token context | 8076 | custom | 38390.00 → 33120.00 ms (-13.7%) | 123.65 → 128.11 ms (+3.6%) | IDENTICAL | matched |
| 3B | Approximately 8K-token context | 8076 | cublas | 22760.00 → 17360.00 ms (-23.7%) | 98.34 → 102.14 ms (+3.9%) | IDENTICAL | matched |
| 7B | Approximately 1K-token context | 1006 | custom | 5630.00 → 5610.00 ms (-0.4%) | 113.59 → 116.32 ms (+2.4%) | IDENTICAL | matched |
| 7B | Approximately 1K-token context | 1006 | cublas | 698.78 → 719.22 ms (+2.9%) | 56.39 → 56.96 ms (+1.0%) | IDENTICAL | matched |
| 7B | Approximately 4K-token context | 4036 | custom | 25800.00 → 25660.00 ms (-0.5%) | 1940.00 → 1970.00 ms (+1.5%) | IDENTICAL | matched |
| 7B | Approximately 4K-token context | 4036 | cublas | 6470.00 → 6450.00 ms (-0.3%) | 1050.00 → 1080.00 ms (+2.9%) | IDENTICAL | matched |
| 7B | Approximately 8K-token context | 8076 | custom | 70460.00 → 62750.00 ms (-10.9%) | 2310.00 → 2360.00 ms (+2.2%) | IDENTICAL | matched |
| 7B | Approximately 8K-token context | 8076 | cublas | 31850.00 → 24250.00 ms (-23.9%) | 1410.00 → 1460.00 ms (+3.5%) | IDENTICAL | matched |
| 14B | Approximately 1K-token context | 1006 | custom | 6540.00 → 6580.00 ms (+0.6%) | 120.14 → 120.34 ms (+0.2%) | IDENTICAL | matched |
| 14B | Approximately 1K-token context | 1006 | cublas | 812.26 → 835.05 ms (+2.8%) | 56.25 → 55.82 ms (-0.8%) | IDENTICAL | matched |
| 14B | Approximately 4K-token context | 4036 | custom | — | — | — | OOM on both revisions |
| 14B | Approximately 4K-token context | 4036 | cublas | — | — | — | not reached after earlier OOM |
| 14B | Approximately 8K-token context | — | custom | — | — | — | not reached after earlier OOM |
| 14B | Approximately 8K-token context | — | cublas | — | — | — | not reached after earlier OOM |
