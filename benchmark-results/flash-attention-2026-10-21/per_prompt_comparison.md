# FlashAttention per-prompt comparison

Each timing cell is previous attention → FlashAttention in milliseconds, with the percentage change in parentheses. Negative percentages are faster.

Previous logs: `benchmark-results/kv-cache-facade-2026-10-20/`. Current raw logs are in this directory. Both use `--max-tokens 16 --linear both --weight-format fp16`; current runs used GPU 4 for all four checkpoints, while the earlier 1.5B, 3B, and 7B runs used GPU 1 and 14B used GPU 4. Both revisions used the single-pass prompt timing path with default GEMM prefill. These are per-prompt measurements, not repeated medians.

The generated output text matched the previous log in all 72 model/prompt/backend rows. The current logs report identical generated tokens across custom and cuBLAS backends and zero non-finite logits in every prompt case.

| Model | Prompt | Input tokens | Custom prefill ms | Custom decode ms | cuBLAS prefill ms | cuBLAS decode ms | Output text |
|---|---|---:|---:|---:|---:|---:|---|
| 1.5B | 1 - Tiny context | 33 | 105.74 → 111.75 (+5.7%) | 231.41 → 232.20 (+0.3%) | 13.74 → 14.47 (+5.3%) | 118.48 → 119.26 (+0.7%) | IDENTICAL |
| 1.5B | 2 - Short context | 45 | 108.02 → 108.10 (+0.1%) | 272.56 → 268.46 (-1.5%) | 13.36 → 13.54 (+1.3%) | 136.55 → 137.73 (+0.9%) | IDENTICAL |
| 1.5B | 3 - Medium context | 154 | 154.69 → 155.55 (+0.6%) | 277.20 → 274.76 (-0.9%) | 23.85 → 24.58 (+3.1%) | 143.02 → 144.45 (+1.0%) | IDENTICAL |
| 1.5B | 4 - Long context | 321 | 251.18 → 253.21 (+0.8%) | 281.84 → 283.05 (+0.4%) | 36.77 → 36.99 (+0.6%) | 153.98 → 152.10 (-1.2%) | IDENTICAL |
| 1.5B | 5 - Long context with repeated structure | 282 | 226.19 → 225.59 (-0.3%) | 279.44 → 280.78 (+0.5%) | 34.89 → 34.73 (-0.5%) | 149.36 → 151.22 (+1.2%) | IDENTICAL |
| 1.5B | 6 - Multi-step arithmetic | 56 | 108.64 → 108.89 (+0.2%) | 272.48 → 269.46 (-1.1%) | 13.72 → 14.09 (+2.7%) | 138.40 → 148.06 (+7.0%) | IDENTICAL |
| 1.5B | 7 - Compare totals | 73 | 118.34 → 118.48 (+0.1%) | 273.11 → 270.08 (-1.1%) | 16.96 → 16.99 (+0.2%) | 142.53 → 147.15 (+3.2%) | IDENTICAL |
| 1.5B | 8 - Exact format | 73 | 118.31 → 118.46 (+0.1%) | 269.16 → 269.60 (+0.2%) | 16.87 → 16.88 (+0.1%) | 139.14 → 145.28 (+4.4%) | IDENTICAL |
| 1.5B | 9 - Time extraction | 78 | 119.48 → 118.77 (-0.6%) | 272.45 → 270.68 (-0.6%) | 17.05 → 17.41 (+2.1%) | 139.25 → 143.25 (+2.9%) | IDENTICAL |
| 3B | 1 - Tiny context | 33 | 188.28 → 188.53 (+0.1%) | 338.35 → 335.50 (-0.8%) | 22.03 → 22.31 (+1.3%) | 159.91 → 159.87 (-0.0%) | IDENTICAL |
| 3B | 2 - Short context | 45 | 192.20 → 192.00 (-0.1%) | 460.97 → 458.95 (-0.4%) | 22.47 → 22.93 (+2.0%) | 217.94 → 219.90 (+0.9%) | IDENTICAL |
| 3B | 3 - Medium context | 154 | 254.96 → 255.14 (+0.1%) | 468.64 → 466.21 (-0.5%) | 39.85 → 40.54 (+1.7%) | 225.18 → 226.66 (+0.7%) | IDENTICAL |
| 3B | 4 - Long context | 321 | 488.56 → 488.70 (+0.0%) | 475.47 → 478.28 (+0.6%) | 61.00 → 60.30 (-1.1%) | 240.25 → 237.28 (-1.2%) | IDENTICAL |
| 3B | 5 - Long context with repeated structure | 282 | 409.18 → 410.71 (+0.4%) | 476.12 → 474.83 (-0.3%) | 55.83 → 57.17 (+2.4%) | 232.99 → 234.91 (+0.8%) | IDENTICAL |
| 3B | 6 - Multi-step arithmetic | 56 | 193.75 → 193.90 (+0.1%) | 464.67 → 459.32 (-1.2%) | 23.26 → 23.09 (-0.7%) | 218.86 → 220.25 (+0.6%) | IDENTICAL |
| 3B | 7 - Compare totals | 73 | 223.53 → 223.69 (+0.1%) | 463.14 → 460.34 (-0.6%) | 29.56 → 29.76 (+0.7%) | 221.54 → 221.63 (+0.0%) | IDENTICAL |
| 3B | 8 - Exact format | 73 | 222.94 → 223.61 (+0.3%) | 461.40 → 460.05 (-0.3%) | 29.59 → 29.56 (-0.1%) | 220.19 → 220.57 (+0.2%) | IDENTICAL |
| 3B | 9 - Time extraction | 78 | 225.51 → 223.00 (-1.1%) | 462.36 → 460.49 (-0.4%) | 29.94 → 29.90 (-0.1%) | 220.46 → 220.68 (+0.1%) | IDENTICAL |
| 7B | 1 - Tiny context | 33 | 311.67 → 311.06 (-0.2%) | 507.32 → 504.25 (-0.6%) | 38.16 → 38.17 (+0.0%) | 230.72 → 230.26 (-0.2%) | IDENTICAL |
| 7B | 2 - Short context | 45 | 322.69 → 326.40 (+1.1%) | 846.42 → 841.04 (-0.6%) | 40.96 → 39.42 (-3.8%) | 385.61 → 384.62 (-0.3%) | IDENTICAL |
| 7B | 3 - Medium context | 154 | 547.44 → 545.56 (-0.3%) | 854.23 → 846.69 (-0.9%) | 79.14 → 78.27 (-1.1%) | 394.85 → 389.64 (-1.3%) | IDENTICAL |
| 7B | 4 - Long context | 321 | 1110.00 → 1110.00 (+0.0%) | 863.52 → 854.93 (-1.0%) | 106.05 → 107.99 (+1.8%) | 401.47 → 398.96 (-0.6%) | IDENTICAL |
| 7B | 5 - Long context with repeated structure | 282 | 963.63 → 961.80 (-0.2%) | 859.30 → 853.31 (-0.7%) | 101.14 → 102.13 (+1.0%) | 400.91 → 396.40 (-1.1%) | IDENTICAL |
| 7B | 6 - Multi-step arithmetic | 56 | 330.58 → 324.99 (-1.7%) | 846.34 → 841.61 (-0.6%) | 41.46 → 40.39 (-2.6%) | 389.59 → 384.78 (-1.2%) | IDENTICAL |
| 7B | 7 - Compare totals | 73 | 414.53 → 414.71 (+0.0%) | 847.86 → 841.29 (-0.8%) | 55.25 → 55.35 (+0.2%) | 389.86 → 385.64 (-1.1%) | IDENTICAL |
| 7B | 8 - Exact format | 73 | 417.98 → 414.78 (-0.8%) | 847.59 → 841.40 (-0.7%) | 55.29 → 55.26 (-0.1%) | 386.98 → 385.48 (-0.4%) | IDENTICAL |
| 7B | 9 - Time extraction | 78 | 416.18 → 415.86 (-0.1%) | 848.44 → 842.30 (-0.7%) | 56.63 → 55.78 (-1.5%) | 389.81 → 385.65 (-1.1%) | IDENTICAL |
| 14B | 1 - Tiny context | 33 | 529.93 → 528.26 (-0.3%) | 1470.00 → 1470.00 (+0.0%) | 79.43 → 79.79 (+0.5%) | 632.73 → 631.19 (-0.2%) | IDENTICAL |
| 14B | 2 - Short context | 45 | 541.19 → 541.17 (-0.0%) | 1700.00 → 1700.00 (+0.0%) | 82.68 → 83.16 (+0.6%) | 732.89 → 728.55 (-0.6%) | IDENTICAL |
| 14B | 3 - Medium context | 154 | 1180.00 → 1180.00 (+0.0%) | 1710.00 → 1710.00 (+0.0%) | 153.88 → 155.37 (+1.0%) | 740.31 → 739.41 (-0.1%) | IDENTICAL |
| 14B | 4 - Long context | 321 | 2150.00 → 2150.00 (+0.0%) | 1720.00 → 1720.00 (+0.0%) | 224.88 → 231.10 (+2.8%) | 754.31 → 754.28 (-0.0%) | IDENTICAL |
| 14B | 5 - Long context with repeated structure | 282 | 2020.00 → 2020.00 (+0.0%) | 1720.00 → 1720.00 (+0.0%) | 209.57 → 211.72 (+1.0%) | 752.18 → 749.85 (-0.3%) | IDENTICAL |
| 14B | 6 - Multi-step arithmetic | 56 | 546.01 → 543.75 (-0.4%) | 1700.00 → 1700.00 (+0.0%) | 85.07 → 85.38 (+0.4%) | 731.90 → 730.22 (-0.2%) | IDENTICAL |
| 14B | 7 - Compare totals | 73 | 843.41 → 843.61 (+0.0%) | 1700.00 → 1700.00 (+0.0%) | 120.43 → 120.82 (+0.3%) | 734.65 → 731.17 (-0.5%) | IDENTICAL |
| 14B | 8 - Exact format | 73 | 844.17 → 843.22 (-0.1%) | 1700.00 → 1700.00 (+0.0%) | 120.16 → 120.66 (+0.4%) | 734.25 → 731.75 (-0.3%) | IDENTICAL |
| 14B | 9 - Time extraction | 78 | 845.37 → 845.73 (+0.0%) | 1590.00 → 1590.00 (+0.0%) | 121.33 → 122.22 (+0.7%) | 684.63 → 682.47 (-0.3%) | IDENTICAL |
