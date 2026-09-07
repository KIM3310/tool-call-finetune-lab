.PHONY: proof check-python install data train eval quantize serve pipeline test lint format format-check typecheck check clean verify help install-gpu merge smoke-test test-cov deploy-cloudflare-pages

PYTHON_MIN_VERSION := 3.10
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
PYTHON_CANDIDATES = $(VENV_PYTHON) python3.13 python3.12 python3.11 python3.10 python3
PYTHON ?= $(shell for py in $(PYTHON_CANDIDATES); do \
	if command -v $$py >/dev/null 2>&1 && $$py -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then \
		command -v $$py; \
		break; \
	fi; \
done)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

check-python:
	@if [ -z "$(PYTHON)" ]; then \
		echo "Python $(PYTHON_MIN_VERSION)+ is required." >&2; \
		echo "Install Python $(PYTHON_MIN_VERSION)+ or run: make PYTHON=/path/to/python$(PYTHON_MIN_VERSION) <target>" >&2; \
		exit 1; \
	fi
	@$(PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || { \
		echo "PYTHON=$(PYTHON) is not Python $(PYTHON_MIN_VERSION)+." >&2; \
		exit 1; \
	}

install: check-python ## Install package in editable mode with dev dependencies
	@if [ ! -x "$(VENV_PYTHON)" ] || ! $(VENV_PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then \
		rm -rf $(VENV); \
		$(PYTHON) -m venv $(VENV); \
	fi
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e ".[dev]"

install-gpu: check-python ## Install with all GPU dependencies (train + quantize + serve)
	@if [ ! -x "$(VENV_PYTHON)" ] || ! $(VENV_PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then \
		rm -rf $(VENV); \
		$(PYTHON) -m venv $(VENV); \
	fi
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

pipeline: ## Run each dependent pipeline stage sequentially, including under make -j
	$(MAKE) data
	$(MAKE) train
	$(MAKE) merge
	$(MAKE) eval
	$(MAKE) quantize

test: ## Run unit tests
	$(VENV_PYTHON) -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	$(VENV_PYTHON) -m pytest tests/ -v --tb=short --cov=tool_call_finetune_lab --cov-report=term-missing

deploy-cloudflare-pages: ## Deploy the static site directory to Cloudflare Pages
	npx --yes wrangler@latest pages deploy site --project-name tool-call-finetune-lab

lint: ## Run linter (ruff)
	$(VENV_PYTHON) -m ruff check src/ tests/

format-check: ## Check formatting without modifying files
	$(VENV_PYTHON) -m ruff format --check src/ tests/

format: ## Auto-format code (ruff)
	$(VENV_PYTHON) -m ruff format src/ tests/
	$(VENV_PYTHON) -m ruff check --fix src/ tests/

typecheck: ## Run type checker (mypy)
	$(VENV_PYTHON) -m mypy src/tool_call_finetune_lab/ --ignore-missing-imports

check: lint format-check typecheck test ## Run all checks (lint + format + typecheck + test)

proof: ## Reproduce synthetic evaluator and data-split contracts without a GPU
	$(VENV_PYTHON) -m tool_call_finetune_lab.eval.contract_probe

verify: install ## Install first, then run checks and reproduce committed contract evidence
	$(MAKE) check
	$(VENV_PYTHON) -m tool_call_finetune_lab.eval.contract_probe --check
	@echo "Verification complete."

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .ruff_cache/ .mypy_cache/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
