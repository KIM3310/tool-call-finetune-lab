# Training Log

A narrative record of the design decisions, trade-offs, and observations made during the development of the QLoRA fine-tuning pipeline for Qwen2.5-7B-Instruct.

## 1. Why QLoRA Over Full Fine-Tuning

Full fine-tuning of a 7B parameter model requires roughly 56 GB of optimizer states and gradients in fp16 (with AdamW, each parameter needs the weight, gradient, and two momentum buffers). That exceeds what a single T4 (16 GB VRAM) or even an A100 40 GB can comfortably handle without aggressive CPU offloading.

QLoRA solves this by:

1. **4-bit NF4 quantization** of the frozen base weights, reducing the 7B model footprint from ~14 GB (fp16) to ~3.5 GB.
2. **LoRA adapters** in fp16 on top, adding only ~26M trainable parameters (0.37% of the base model).
3. **Paged AdamW** via bitsandbytes, which offloads optimizer state pages to CPU RAM when GPU memory is full.

On a Kaggle T4 with 16 GB VRAM, this combination fits comfortably with gradient checkpointing enabled, leaving enough headroom for a batch size of 1 with gradient accumulation of 8 (effective batch size of 8).

The practical trade-off: QLoRA converges slightly slower than full fine-tuning due to the quantization noise in the frozen weights, and the final model quality has a small gap compared to full fine-tuning on the same data. For our use case (improving structured output formatting, not teaching new world knowledge), this gap is acceptable.

## 2. LoRA Rank Selection: Why rank=16

LoRA rank controls the capacity of the adapter. Higher rank means the adapter can express more complex weight deltas, but also means more parameters to fit, higher memory usage, and greater overfitting risk on small datasets.

Ranks tested during initial experiments:

| Rank | Trainable Params | Val Loss (500 steps) | Notes |
|------|-----------------|---------------------|-------|
| 4    | ~6.5M           | 0.82                | Underfitting on parallel-call examples |
| 8    | ~13M            | 0.71                | Reasonable, but simple-call accuracy plateaued early |
| 16   | ~26M            | 0.64                | Best val loss without signs of overfitting |
| 32   | ~52M            | 0.63                | Marginally better loss, but training loss diverged from val loss by step 800, suggesting overfitting |

Rank 16 with alpha=32 (scaling factor of 2.0) provided the best balance. The alpha-to-rank ratio of 2:1 is a common heuristic from the original LoRA paper that keeps the adapter's effective learning rate stable.

Target modules were set to `q_proj`, `k_proj`, `v_proj`, and `o_proj` (all attention projections). We did not include MLP layers (`gate_proj`, `up_proj`, `down_proj`) because initial tests showed that attention-only targeting was sufficient for the structured output formatting task, and including MLP layers doubled the trainable parameters without meaningful improvement in tool-call accuracy.

## 3. Why 1 Epoch

The training data consists of 23,716 examples. At an effective batch size of 8, that is roughly 2,965 optimization steps per epoch.

Key observations from W&B metrics:

- **Training loss** dropped steeply in the first 500 steps (from ~1.8 to ~0.7), then gradually decreased to ~0.45 by end of epoch 1.
- **Validation loss** tracked training loss closely through epoch 1, reaching ~0.52 at step 2,900.
- In a preliminary 3-epoch run, validation loss began increasing at step ~3,500 (mid-epoch 2), a clear signal of overfitting.

One epoch is a deliberate choice to prevent catastrophic forgetting. The base Qwen2.5-7B-Instruct model already has strong general instruction-following ability. Over-training on tool-call-specific data risks degrading its performance on non-tool prompts, which matters because in production the same model handles both tool calls and regular conversation through stage-pilot.

The cosine learning rate schedule with 10% warmup (roughly 296 warmup steps) ensures the learning rate peaks and then decays smoothly within the single epoch, giving the model a complete annealing cycle.

## 4. Data Composition Decisions

The training data merges two sources:

- **BFCL v4** (2,501 examples): High-quality, benchmark-grade tool-calling examples with ground-truth answers. These cover the exact distribution we evaluate against, but the dataset is small.
- **Glaive v2** (subsampled to ~27K after dedup): Multi-turn function-calling conversations. More diverse and conversational, but noisier. Some examples have inconsistent tool schemas or unusual formatting.

Deduplication by SHA-256 content hash removed ~1,100 near-duplicate examples from Glaive. Stratified splitting ensured each (source, category) combination was proportionally represented in train/val/test.

The rationale for mixing both sources: BFCL alone is too small to fine-tune effectively (overfits within 200 steps), while Glaive alone teaches general function-calling patterns but does not align the model specifically with BFCL's expected output format. The combination gives the model both volume and format precision.

## 5. W&B Metrics and What They Showed

Key metrics tracked during training:

- **train/loss**: Smooth descent from 1.8 to 0.45 over one epoch. No loss spikes, suggesting data quality was adequate and the learning rate was appropriate.
- **eval/loss**: Closely tracked training loss (gap < 0.08), confirming the model was not overfitting at 1 epoch.
- **train/learning_rate**: Cosine schedule peaked at 2e-4 around step 296, decayed to near-zero by step 2,965.
- **train/grad_norm**: Stayed below 1.0 throughout (gradient clipping set to max_grad_norm=1.0). No gradient explosion events.
- **system/gpu_memory_allocated**: Peaked at ~14.2 GB on T4, leaving ~1.8 GB headroom. Gradient checkpointing was essential for this to work; without it, peak memory exceeded 16 GB and caused OOM.

## 6. Gradient Checkpointing Trade-off

Enabling gradient checkpointing reduced peak GPU memory by roughly 40% (from ~23 GB theoretical to ~14.2 GB actual) at the cost of approximately 30% slower training throughput. On a Kaggle T4, each epoch took approximately 4.5 hours with checkpointing enabled versus an estimated 3.5 hours without (which would not fit in memory anyway).

This trade-off is necessary on consumer and free-tier hardware. On an A100 80 GB, gradient checkpointing could be disabled for faster iteration.

## 7. Post-Training Steps Completed

1. **BFCL evaluation path**: The repo contains the full BFCL runner and comparison path. The currently checked-in Kaggle notebook demonstrates a smaller first-100-example smoke eval, while the full strict artifact should be treated as pending until `results/bfcl_results.json` is republished.
2. **AWQ quantization**: Quantized the merged model to INT4 with AWQ for efficient vLLM serving.
3. **Inference benchmarking**: Measured tokens/second on T4 and A100, confirming the quantized model meets latency requirements for stage-pilot integration.
4. **External artifact links**: Hugging Face and W&B references are tracked in the README, but their public accessibility should be reverified before using them as recruiter-facing proof.
