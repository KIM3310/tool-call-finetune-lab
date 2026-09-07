# Tool-Call Fine-Tune Lab: design and evidence

Updated 2026-09-07.

## Design decision

Evaluation predicts a target from prior conversation only, compares full JSON arguments and groups identical requests across data partitions. Dataset and scorer fingerprints must match before reports can be compared.

## Inspect the code

- [src/tool_call_finetune_lab/eval/bfcl_runner.py](../src/tool_call_finetune_lab/eval/bfcl_runner.py): Context and label integrity in evaluation.
- [tests/test_evaluation_integrity.py](../tests/test_evaluation_integrity.py): Leakage, split and structured argument regressions.

## Scope of the evidence

CPU contract evidence uses synthetic data. No GPU training run, checkpoint-quality gain or official BFCL score is claimed.

## Contribution and provenance

These notes describe what can be inspected in the repository. Commit history and pull-request diffs preserve the change trail; they do not independently establish manual versus AI-assisted authorship, team roles or contribution percentages. No such percentages are inferred here.

[Project overview](../README.md)
