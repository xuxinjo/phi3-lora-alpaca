"""
Merge LoRA adapter weights into the base Phi-3 Mini model.

Produces a standalone HuggingFace model loadable with
AutoModelForCausalLM.from_pretrained() without PEFT.
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src import config
from src.models.load_model import _ensure_rope_scaling

DEFAULT_BASE_MODEL = config.PHI3_MODEL_ID
DEFAULT_LORA_PATH = "checkpoints/lora_phi3"
DEFAULT_OUTPUT_PATH = "merged_model/phi3_alpaca_lora_merged"


def merge_lora(
    base_model: str,
    lora_path: str,
    output_path: str,
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
) -> None:
    """
    Load base Phi-3, apply LoRA adapter, merge, and save as standalone model.

    Args:
        base_model: HuggingFace model ID or path to base model.
        lora_path: Path to LoRA adapter directory.
        output_path: Directory to save merged model.
        torch_dtype: Model dtype (fp16 or fp32). Defaults to bfloat16 if available else float32.
        trust_remote_code: Whether to trust custom model code.
    """
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("Loading base model...")
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    _ensure_rope_scaling(config)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(model, lora_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print("Saving merged model...")
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base Phi-3 Mini model",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model ID or path (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=DEFAULT_LORA_PATH,
        help=f"Path to LoRA adapter directory (default: {DEFAULT_LORA_PATH})",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output directory for merged model (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use float32 instead of bfloat16",
    )
    args = parser.parse_args()

    if not Path(args.lora_path).exists():
        raise FileNotFoundError(
            f"LoRA path not found: {args.lora_path}. "
            "Train first with: python main.py train"
        )

    dtype = torch.float32 if args.fp32 else None
    merge_lora(
        base_model=args.base_model,
        lora_path=args.lora_path,
        output_path=args.output_path,
        torch_dtype=dtype,
    )


if __name__ == "__main__":
    main()
