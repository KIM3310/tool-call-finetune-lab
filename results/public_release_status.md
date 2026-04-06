# Public Release Status

- Generated on: `2026-03-29`
- Data snapshot: `23716` train / `2962` val / `2969` test

## Kaggle

- Public dataset page probe: `200`
- Kernel page probe: `200`
- Latest manual publish check: `succeeded`
- Latest observed kernel runtime status: `COMPLETE`
- Checked-in Kaggle smoke result mirror: `results/kaggle_public_smoke_bfcl_results.json`

## Hugging Face

- LoRA repo probe: `401`
- AWQ repo probe: `401`

## Interpretation

- The attached Kaggle dataset page is public and the Kaggle notebook page is now live.
- The latest authenticated Kaggle push succeeded, and the remote kernel reached `COMPLETE` on Kaggle.
- That successful public run used the smoke fallback path on an unsupported accelerator, so it proves public execution hygiene rather than full QLoRA benchmark output.
- The repo mirrors the completed public Kaggle smoke result as a checked-in JSON artifact.
- The Hugging Face artifact URLs are not publicly reachable from this environment right now.
- A full `results/bfcl_results.json` still requires the actual fine-tuned weights to be available again.
