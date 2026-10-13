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

Weights are currently converted to f32 before upload. This means the model
parameters alone use roughly 4 bytes each on the GPU: the tested 3.09B model
uses about 11.5 GiB for weights, while a 7B model would be too close to the
32 GiB card limit for the current implementation. Allow additional room for
CUDA allocations and temporary tensors.

## Checkpoints exercised

| Checkpoint | Parameters | Weight layout | V100 run |
| --- | ---: | --- | --- |
| Qwen2.5-0.5B-Instruct | 494M | 1 file | 8-token smoke test |
| Qwen2.5-1.5B-Instruct | 1.54B | 1 file | 8-token smoke test |
| Qwen2.5-3B-Instruct | 3.09B | 2 shards | 8-token smoke test |
