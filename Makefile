.PHONY: install data train eval quantize serve pipeline test lint format

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) -m tool_call_finetune_lab.data.prepare_bfcl
	$(PYTHON) -m tool_call_finetune_lab.data.prepare_glaive
	$(PYTHON) -m tool_call_finetune_lab.data.merge_and_split

train:
	$(PYTHON) -m tool_call_finetune_lab.train.lora_trainer

eval:
	$(PYTHON) -m tool_call_finetune_lab.eval.bfcl_runner
	$(PYTHON) -m tool_call_finetune_lab.eval.compare

quantize:
	$(PYTHON) -m tool_call_finetune_lab.quantize.awq_quantize

serve:
	$(PYTHON) -m tool_call_finetune_lab.serve.vllm_launcher

pipeline: data train eval quantize
	@echo "Full pipeline complete."

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m ruff format src/ tests/
