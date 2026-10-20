# Tested checkpoints

The engine was extended and exercised with Qwen2.5 instruct checkpoints close
to the requested 8B and 13B sizes. The available official sizes are 7B and
14B, so those are the tested substitutes. Both repositories declare Apache
2.0 licensing.

| Local directory | Hugging Face repository | Pinned revision | Parameters | Source weights |
|---|---|---|---:|---:|
| `models/Qwen2.5-7B-Instruct` | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | `a09a35458c702b33eeacc393d103063234e8bc28` | 7,615,616,512 | 15 GB |
| `models/Qwen2.5-14B-Instruct` | [`Qwen/Qwen2.5-14B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` | 14,770,033,664 | 28 GB |

Download the exact tested snapshots with the Hugging Face CLI:

```bash
hf download Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --include '*.json' --include 'model-*.safetensors' \
  --local-dir models/Qwen2.5-7B-Instruct

hf download Qwen/Qwen2.5-14B-Instruct \
  --revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --include '*.json' --include 'model-*.safetensors' \
  --local-dir models/Qwen2.5-14B-Instruct
```

The checkpoint files are intentionally not stored in Git: `models/` is
gitignored and the two snapshots take about 43 GB. In the benchmark workspace,
the 7B files are under `models/`; the 14B path is a symlink to
`/tmp/InferenceEngine-models/Qwen2.5-14B-Instruct` because of the home-volume
quota. The commands above put both under `models/` on a machine with enough
space.

To validate a downloaded snapshot and run the full nine-prompt suite on both
linear backends, use:

```bash
cargo run --release -- --gpu 1 --model-dir models/Qwen2.5-7B-Instruct \
  --max-tokens 16 --linear both --weight-format fp16
cargo run --release -- --gpu 1 --model-dir models/Qwen2.5-14B-Instruct \
  --max-tokens 16 --linear both --weight-format fp16
```

Change `--weight-format` to `f32` or `int8` for the other storage formats.
`BASE_PROMPTS.txt` is the source of the nine cases. Full captured outputs and
per-prompt timings are archived in
[`benchmark-results/precision-2026-10-20/`](benchmark-results/precision-2026-10-20/)
and summarized in [`HISTORY_BENCHMARK.md`](HISTORY_BENCHMARK.md).
