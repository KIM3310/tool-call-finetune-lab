"""Push model + model card to HuggingFace Hub."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from textwrap import dedent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_CARD_TEMPLATE = dedent(
    """\
    ---
    language:
    - en
    license: apache-2.0
    base_model: Qwen/Qwen2.5-7B-Instruct
    tags:
    - tool-calling
    - function-calling
    - lora
    - qwen2.5
    - qlora
    datasets:
    - gorilla-llm/berkeley-function-call-leaderboard
    - glaiveai/glaive-function-calling-v2
    pipeline_tag: text-generation
    ---

    # {repo_id}

    Fine-tuned **Qwen2.5-7B-Instruct** for reliable tool-calling via QLoRA (rank=16) on
    a mixed BFCL + Glaive-v2 corpus.

    ## Usage

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
    ```

    Or with vLLM:
    ```bash
    python -m vllm.entrypoints.openai.api_server \\
        --model {repo_id} \\
        --tool-call-parser hermes \\
        --port 8000
    ```

    ## Training Details

    - **Base model**: `Qwen/Qwen2.5-7B-Instruct`
    - **Method**: QLoRA (4-bit NF4), rank=16, alpha=32
    - **Data**: BFCL + Glaive-function-calling-v2
    - **Epochs**: 1, lr=2e-4

    ## Evaluation (BFCL)

    | Category | Accuracy |
    |---|---|
    | AST Simple | — |
    | AST Multiple | — |
    | AST Parallel | — |
    | Overall | — |

    {extra}

    ## Source

    Training code: [KIM3310/tool-call-finetune-lab](https://github.com/KIM3310/tool-call-finetune-lab)

    ## License

    Apache-2.0
    """
)


def _write_model_card(model_path: str, repo_id: str, extra: str = "") -> str:
    """Write model card README.md into the model dir."""
    card_content = MODEL_CARD_TEMPLATE.format(repo_id=repo_id, extra=extra)
    card_path = Path(model_path) / "README.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card_content)
    logger.info("Model card written to %s", card_path)
    return str(card_path)


def push_to_hub(
    model_path: str,
    repo_id: str,
    private: bool = False,
    model_card_extra: str = "",
    commit_message: str = "Upload fine-tuned model",
) -> str:
    """Upload model dir to HF Hub. Returns the repo URL."""
    from huggingface_hub import HfApi, create_repo

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise OSError(
            "HF_TOKEN environment variable is required for pushing to Hub. "
            "Set it in .env or export HF_TOKEN=your_token"
        )

    api = HfApi(token=hf_token)

    # Create repo if it doesn't exist
    logger.info("Creating/accessing repository: %s", repo_id)
    repo_url = create_repo(
        repo_id=repo_id,
        token=hf_token,
        private=private,
        exist_ok=True,
    )
    logger.info("Repository URL: %s", repo_url)

    # Write model card
    _write_model_card(model_path, repo_id, model_card_extra)

    # Upload all files in the model directory
    logger.info("Uploading model from %s to %s ...", model_path, repo_id)
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message=commit_message,
        token=hf_token,
        ignore_patterns=["*.pyc", "__pycache__", ".git", "wandb", "outputs"],
    )

    model_url = f"https://huggingface.co/{repo_id}"
    logger.info("Upload complete: %s", model_url)

    # Print summary
    print("\nModel pushed successfully!")
    print(f"URL: {model_url}")
    print(f"Load with: AutoModelForCausalLM.from_pretrained('{repo_id}')")

    return model_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push model to HuggingFace Hub")
    parser.add_argument("--model-path", default="outputs/merged-model")
    parser.add_argument("--repo-id", default="KIM3310/qwen2.5-7b-tool-calling-lora")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--model-card-extra", default="")
    parser.add_argument(
        "--commit-message", default="Upload fine-tuned Qwen2.5-7B tool-call model"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    push_to_hub(
        model_path=args.model_path,
        repo_id=args.repo_id,
        private=args.private,
        model_card_extra=args.model_card_extra,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
