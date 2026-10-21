# CUTLASS vendored subset

Upstream: NVIDIA CUTLASS v3.9.2
Commit: `ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e`

The inference engine uses the headers in `include/`, utility headers in
`tools/util/include/`, and the fused multi-head attention implementation in
`examples/41_fused_multi_head_attention/`. The subset is built for SM70 at
project build time and loaded as PTX at runtime. See `LICENSE.txt` for the
upstream BSD-3-Clause license.

Local source patch: the single-query MQA shortcut in `kernel_forward.h` is
disabled when a custom batch hook is active. The CUTLASS wrapper's hook maps
each Qwen query head to its GQA KV head, so the shortcut's assumption that all
heads share one K/V pointer is not valid for that configuration.
