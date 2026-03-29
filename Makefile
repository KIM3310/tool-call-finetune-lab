.PHONY: install data train eval quantize serve pipeline test lint format typecheck check clean verify help install-gpu merge smoke-test test-cov

PYTHON ?= python3.11
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode with dev dependencies
	test -x "$(VENV_PYTHON)" || $(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e ".[dev]"

install-gpu: ## Install with all GPU dependencies (train + quantize + serve)
	test -x "$(VENV_PYTHON)" || $(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e ".[gpu,dev]"

data: ## Download and prepare training data (no GPU needed)
	$(VENV_PYTHON) -m tool_call_finetune_lab.data.prepare_bfcl
	$(VENV_PYTHON) -m tool_call_finetune_lab.data.prepare_glaive
	$(VENV_PYTHON) -m tool_call_finetune_lab.data.merge_and_split

train: ## Run QLoRA fine-tuning (requires GPU)
	$(VENV_PYTHON) -m tool_call_finetune_lab.train.lora_trainer

merge: ## Merge LoRA adapter into base model
	$(VENV_PYTHON) -m tool_call_finetune_lab.train.merge_adapter

eval: ## Run BFCL evaluation and generate comparison table
	$(VENV_PYTHON) -m tool_call_finetune_lab.eval.bfcl_runner
	$(VENV_PYTHON) -m tool_call_finetune_lab.eval.compare

quantize: ## AWQ INT4 quantization of the merged model
	$(VENV_PYTHON) -m tool_call_finetune_lab.quantize.awq_quantize

serve: ## Launch vLLM server with the quantized model
	$(VENV_PYTHON) -m tool_call_finetune_lab.serve.vllm_launcher

smoke-test: ## Run smoke tests against a running vLLM server
	$(VENV_PYTHON) -m tool_call_finetune_lab.serve.openai_compat_test

pipeline: data train merge eval quantize ## Run the full pipeline (data -> train -> merge -> eval -> quantize)
	@echo "Full pipeline complete."

test: ## Run unit tests
	$(VENV_PYTHON) -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	$(VENV_PYTHON) -m pytest tests/ -v --tb=short --cov=tool_call_finetune_lab --cov-report=term-missing

lint: ## Run linter (ruff)
	$(VENV_PYTHON) -m ruff check src/ tests/

format: ## Auto-format code (ruff)
	$(VENV_PYTHON) -m ruff format src/ tests/
	$(VENV_PYTHON) -m ruff check --fix src/ tests/

typecheck: ## Run type checker (mypy)
	$(VENV_PYTHON) -m mypy src/tool_call_finetune_lab/ --ignore-missing-imports

check: lint typecheck test ## Run all checks (lint + typecheck + test)

verify: install test lint ## Run verification (self-bootstrap + test + lint)
	@echo "Verification complete."

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .ruff_cache/ .mypy_cache/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
