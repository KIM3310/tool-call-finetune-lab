# Contributing to Tool-Call Fine-Tune Lab

Thanks for your interest in contributing. This guide covers the development workflow, coding standards, and how to submit changes.

## Prerequisites

- Python 3.10+
- A working `pip` installation (virtualenv or conda recommended)
- Git
- For training-related contributions: a CUDA-capable GPU with PyTorch 2.1+

## Development Setup

```bash
# Clone the repo
git clone https://github.com/KIM3310/tool-call-finetune-lab.git
cd tool-call-finetune-lab

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# If working on training or eval code, also install GPU deps
pip install -e ".[gpu]"
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

- Line length: 100 characters
- Target version: Python 3.10
- Enabled rules: `E`, `F`, `I`, `UP` (pyflakes, pycodestyle, isort, pyupgrade)

Run the linter before committing:

```bash
python -m ruff check src/ tests/
python -m ruff format src/ tests/
```

## Type Checking

We use [mypy](https://mypy.readthedocs.io/) for static type analysis:

```bash
python -m mypy src/
```

All public functions should have type annotations. Use `from __future__ import annotations` at the top of each module.

## Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_data_pipeline.py -v

# Run with coverage (if installed)
python -m pytest tests/ --cov=src/tool_call_finetune_lab
```

Tests should not require a GPU or network access. Mock external dependencies (HuggingFace Hub, W&B, OpenAI API) in tests.

## Project Structure

```
src/tool_call_finetune_lab/
  config.py        # Dataclass configs for all stages
  data/            # Data download, parsing, merging, splitting
  train/           # QLoRA trainer and adapter merge
  eval/            # BFCL runner and comparison utilities
  quantize/        # AWQ quantization and benchmarking
  serve/           # vLLM launcher and smoke tests
tests/             # Mirrors src/ structure
```

## Making Changes

### Branching

1. Fork the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Keep branches focused on a single change. Separate unrelated fixes into different PRs.

### Commit Messages

Use the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
feat: add beam search decoding option to eval runner
fix: handle empty tool_calls array in Glaive parser
docs: add QLoRA hyperparameter rationale to training log
test: add edge case tests for parallel tool-call formatting
```

### Pull Requests

1. Ensure all tests pass and the linter reports no issues.
2. Add tests for new functionality. Aim to cover both the happy path and edge cases.
3. Update documentation if you change public APIs or configuration options.
4. Fill in the PR template with a description of what changed and why.

## Areas Where Contributions Are Welcome

- **Evaluation**: Adding new eval categories or improving BFCL result parsing
- **Data pipelines**: Supporting additional function-calling datasets
- **Serving**: Optimizing vLLM configuration or adding new serving backends
- **Tests**: Increasing coverage, especially for edge cases in data formatting
- **Documentation**: Improving training guides, adding architecture diagrams

## Reporting Issues

Use the GitHub issue templates:

- **Bug report**: For errors, crashes, or incorrect behavior
- **Feature request**: For new capabilities or improvements

Include reproduction steps, environment details (Python version, GPU, OS), and relevant logs.

## License

By contributing, you agree that your contributions will be licensed under the [Apache-2.0 License](LICENSE).
