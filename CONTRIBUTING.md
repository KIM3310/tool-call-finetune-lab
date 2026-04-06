# Contributing

## Setup

```bash
git clone https://github.com/KIM3310/tool-call-finetune-lab.git
cd tool-call-finetune-lab
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# for training/eval work:
pip install -e ".[gpu]"
```

## Code Style

[Ruff](https://docs.astral.sh/ruff/) for linting + formatting. Line length 100, Python 3.10 target.

```bash
python -m ruff check src/ tests/
python -m ruff format src/ tests/
```

## Type Checking

```bash
python -m mypy src/
```

Public functions should have type annotations. Use `from __future__ import annotations`.

## Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src/tool_call_finetune_lab
```

Tests shouldn't need a GPU or network. Mock external deps (HF Hub, W&B, OpenAI).

## Making Changes

1. Fork + branch from `main`
2. [Conventional Commits](https://www.conventionalcommits.org/) format: `feat:`, `fix:`, `docs:`, `test:`
3. Tests pass, linter clean
4. Add tests for new stuff
5. Update docs if you change public APIs or config

## Areas for Contribution

- Eval: new categories, better BFCL result parsing
- Data: support more function-calling datasets
- Serving: vLLM config optimization, new backends
- Tests: edge case coverage, especially data formatting
- Docs: training guides, diagrams

## Issues

Use the GitHub templates (bug report / feature request). Include repro steps, env details, logs.

## License

Contributions are Apache-2.0 licensed (see [LICENSE](LICENSE)).
