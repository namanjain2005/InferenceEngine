# Model support

The inference loop uses the `ModelBackend` contract in `src/model.rs`. A backend
provides a shared `ModelConfig`, allocates its own cache type, and implements a
single-token forward pass. `Qwen2Model` is the first backend. It currently reads
Hugging Face Qwen2/Qwen2.5 causal decoder checkpoints with tied token embeddings,
QKV biases, RoPE, and SwiGLU.

The generic interface is intentionally at the decoder boundary. A new model
family still needs its own config mapping, checkpoint tensor mapping, and
backend implementation; sharing the interface does not imply that different
architectures already share the same math.

The forward pass keeps logits on the GPU. A CUDA reduction selects the next
token, and only that token ID is copied to the host. Prompt prefill reduces
only the final prompt position because earlier prompt tokens are already known.

## Checkpoint validation

Download the additional local checkpoints with:

```sh
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct
hf download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct
```

The loader accepts either `model.safetensors` or the standard
`model.safetensors.index.json` plus its shard files. Run the metadata and tensor
shape check without a GPU:

```sh
cargo run -- --check-model --model-dir models/Qwen2.5-1.5B-Instruct
```

Run a short real forward/generation pass on a V100:

```sh
cargo run --release -- --gpu 0 \
  --model-dir models/Qwen2.5-3B-Instruct \
  --prompt 'Reply with a short greeting.' --max-tokens 8
```

Run the full suite in `BASE_PROMPTS.txt` by omitting `--prompt`:

```sh
cargo run --release -- --gpu 0 \
  --model-dir models/Qwen2.5-3B-Instruct --max-tokens 48
```

Weights are currently converted to f32 before upload. This means the model
parameters alone use roughly 4 bytes each on the GPU: the tested 3.09B model
uses about 11.5 GiB for weights, while a 7B model would be too close to the
32 GiB card limit for the current implementation. Allow additional room for
CUDA allocations and temporary tensors.

## Checkpoints exercised

| Checkpoint | Parameters | Weight layout | V100 run |
| --- | ---: | --- | --- |
| Qwen2.5-0.5B-Instruct | 494M | 1 file | 9-prompt suite |
| Qwen2.5-1.5B-Instruct | 1.54B | 1 file | 9-prompt suite |
| Qwen2.5-3B-Instruct | 3.09B | 2 shards | 9-prompt suite |

All suite runs completed without inference errors or OOMs. Text quality was not
uniform: the arithmetic prompt did not produce its requested result within 48
tokens on any size; the comparison prompt was wrong on 0.5B and 3B; and 1.5B
answered the time question with 180 minutes instead of 90. Treat the suite as a
forward-pass and generation smoke test, not a model accuracy benchmark.

## GPU argmax check

On GPU 0, a single 8-token greeting run before and after moving argmax to CUDA
measured:

| Model | Decode before → after | Prefill before → after |
| --- | ---: | ---: |
| 0.5B | 112.11 → 153.76 tok/s | 230.13 → 162.50 ms |
| 1.5B | 65.67 → 75.94 tok/s | 391.87 → 333.82 ms |
| 3B | 40.18 → 43.75 tok/s | 634.36 → 573.59 ms |

These are single short smoke runs, so treat the rates as directional rather than
stable benchmarks. All three generated the same greeting as before.
