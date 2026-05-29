# Review Guide - Tool-Call Fine-Tune Lab

Updated: 2026-05-30

Use this page as the short path through the repository. It keeps the review grounded in the code, docs, commands, and boundaries that are already present.

## Summary

| Field | Notes |
|---|---|
| Lane | B2B model adaptation and eval |
| Core idea | QLoRA/BFCL/tool-call adaptation lab with serving-readiness notes. |
| Primary reader | AI platform teams, applied ML engineers, and model evaluation groups. |
| Stack | Python, Docker |

## Open First

1. Start with the README fast path and architecture section.
2. Open `docs/monetization-playbook.md` only when reviewing the product or service angle.
3. Check the commands below before making claims about quality.
4. Skim the CI workflows and fixture data before deeper implementation review.
5. Read the boundaries section before presenting the project externally.

## Checks

| Purpose | Command |
|---|---|
| Full local gate | `make verify` |
| Test suite | `make test` |

## CI

- .github/workflows/architecture-blueprint.yml
- .github/workflows/ci.yml
- .github/workflows/dependency-review.yml
- .github/workflows/repository-health.yml
- .github/workflows/repository-surface.yml
- .github/workflows/secret-scan.yml

## Evidence

- pytest/ruff-style local verification path
- containerized delivery path
- make verify passes
- Training config is explicit
- GPU paths are separated

## Commercial Notes

| Possible offer | Working price assumption |
|---|---|
| Model adaptation study | $5k-$15k eval study |
| BFCL-style evaluation pack | $15k-$60k adaptation sprint |
| Serving readiness review | $3k-$12k/month model eval ops |

## Boundaries

- Dataset licensing required
- Eval coverage before deployment
- No model-quality guarantees without runs

## Useful Metrics

- Tool-call accuracy
- Eval coverage
- Serving latency posture
