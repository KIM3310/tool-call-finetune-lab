# Training Log

Notes on decisions and things I ran into while building the QLoRA pipeline for Qwen2.5-7B-Instruct.

## 1. Why QLoRA

Full fine-tuning a 7B model needs ~56 GB for optimizer states + gradients in fp16 (AdamW: weight + gradient + 2 momentum buffers per param). Doesn't fit on a T4 (16 GB) or even A100 40 GB without aggressive offloading.

QLoRA fixes this:

1. **4-bit NF4 quantization** of frozen base weights: 7B model goes from ~14 GB (fp16) to ~3.5 GB
2. **LoRA adapters** in fp16 on top: only ~26M trainable params (0.37% of base)
3. **Paged AdamW** via bitsandbytes: offloads optimizer pages to CPU when GPU fills up

On Kaggle T4 with gradient checkpointing, this fits fine. Batch size 1 with grad accum 8 (effective batch 8).

Tradeoff: QLoRA converges a bit slower than full FT due to quantization noise, and there's a small quality gap. For our task (structured output formatting, not new knowledge), it's fine.

## 2. LoRA Rank: why 16

Higher rank = more capacity but also more params, more memory, more overfitting risk.

| Rank | Trainable Params | Val Loss (500 steps) | Notes |
|------|-----------------|---------------------|-------|
| 4    | ~6.5M           | 0.82                | Underfitting on parallel-call examples |
| 8    | ~13M            | 0.71                | OK but simple-call accuracy plateaued early |
| 16   | ~26M            | 0.64                | Best val loss, no overfitting |
| 32   | ~52M            | 0.63                | Barely better, train/val diverged by step 800 |

Rank 16, alpha=32 (2:1 ratio from the LoRA paper). Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`. Tried including MLP layers too but it doubled params without helping tool-call accuracy.

## 3. Why 1 Epoch

23,716 training examples. At effective batch 8, that's ~2,965 steps per epoch.

From W&B:
- Train loss: steep drop first 500 steps (1.8 -> 0.7), then gradual to ~0.45 by end of epoch 1
- Val loss tracked close (reached ~0.52 at step 2,900)
- In a 3-epoch test run, val loss started climbing at step ~3,500 (mid-epoch 2) -- overfitting

One epoch is intentional. The base Qwen model already does instruction-following well. Overtraining on tool-call data risks degrading non-tool prompts, and in production the same model handles both (through stage-pilot).

Cosine LR with 10% warmup (~296 steps) gives a full annealing cycle within the single epoch.

## 4. Data Mix

Two sources:

- **BFCL v4** (2,501 examples): benchmark-grade with ground truth. Covers the exact eval distribution but too small on its own (overfits within 200 steps).
- **Glaive v2** (subsampled to ~27K after dedup): multi-turn function-calling convos. More diverse but noisier -- some have inconsistent tool schemas or weird formatting.

SHA-256 dedup removed ~1,100 near-dupes from Glaive. Stratified split by (source, category).

BFCL alone = too small. Glaive alone = teaches general patterns but doesn't align with BFCL's expected output format. Mixing gives volume + format precision.

## 5. W&B Metrics

- **train/loss**: smooth 1.8 -> 0.45, no spikes
- **eval/loss**: tracked train loss (gap < 0.08), no overfitting at 1 epoch
- **learning_rate**: cosine, peaked at 2e-4 ~step 296, decayed to near-zero by 2,965
- **grad_norm**: stayed below 1.0 (clipping at max_grad_norm=1.0), no explosions
- **gpu_memory**: peaked at ~14.2 GB on T4, ~1.8 GB headroom. Without gradient checkpointing it OOMs.

## 6. Gradient Checkpointing

Reduced peak GPU memory ~40% (from ~23 GB theoretical to ~14.2 GB actual). Cost: ~30% slower training. Each epoch took about 4.5h with checkpointing vs estimated 3.5h without (which wouldn't fit in memory anyway).

Necessary on T4/free-tier hardware. On A100 80 GB you could turn it off for faster iteration.

## 7. Post-Training

1. **BFCL eval**: full runner and comparison code in the repo. The Kaggle notebook does a 100-example smoke eval; full strict eval needs `results/bfcl_results.json` republished with actual fine-tuned weights.
2. **AWQ quantization**: merged model -> INT4 for vLLM serving
3. **Inference benchmarking**: measured tokens/sec on T4 and A100, meets latency requirements for stage-pilot
4. **External artifacts**: HF and W&B links in README, but should re-verify public access before sharing

## 8. Kaggle Iteration History

This didn't work on the first try. Roughly 20+ Kaggle T4 sessions before the final config:

- **OOM**: initial runs without gradient checkpointing blew past 16 GB. Had to enable `gradient_checkpointing=True` and reduce batch size.
- **Data format bugs**: early BFCL/Glaive merging produced malformed chat templates that caused silent training degradation. Took several iterations to get `<tool_call>` formatting right.
- **LoRA rank exploration**: tested 4/8/16/32 across multiple sessions (see section 2)
- **Quantization issues**: AWQ initially failed on some layer configs. Had to debug `w_bit` and `q_group_size`.
- **Session timeouts**: Kaggle's 12h GPU limit forced checkpointing intermediate results and resuming across sessions.

Each failure -> root cause -> config fix -> next attempt. The final `make pipeline` runs end-to-end.
