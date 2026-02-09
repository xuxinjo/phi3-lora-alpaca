"""Model loading utilities."""

from typing import Any, Optional

import torch
from peft import prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src import config


PHI3_MODEL_ID = config.PHI3_MODEL_ID


def _ensure_rope_scaling(config: Any) -> None:
    """Normalize rope_scaling config to avoid KeyError in Phi-3 model initialization."""
    current_rs = getattr(config, "rope_scaling", None)
    if current_rs is None:
        return
    if isinstance(current_rs, dict) and current_rs.get("type") is None:
        config.rope_scaling = None
        if hasattr(config, "__dict__"):
            config.__dict__["rope_scaling"] = None
    elif isinstance(current_rs, dict) and hasattr(config, "__dict__"):
        config.__dict__["rope_scaling"] = config.rope_scaling


def load_phi3_4bit(
    model_id: str = PHI3_MODEL_ID,
    bnb_4bit_compute_dtype: Optional[torch.dtype] = None,
    bnb_4bit_quant_type: str = "nf4",
    trust_remote_code: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Phi-3 Mini with 4-bit quantization and prepare it for LoRA fine-tuning.

    Args:
        model_id: HuggingFace model ID (default: Phi-3 Mini 4k instruct).
        bnb_4bit_compute_dtype: Dtype for 4-bit compute. Defaults to bfloat16.
        bnb_4bit_quant_type: Quantization type: "nf4" or "fp4".
        trust_remote_code: Whether to trust custom model code.

    Returns:
        (model, tokenizer) with model ready for LoRA.
    """
    if bnb_4bit_compute_dtype is None:
        bnb_4bit_compute_dtype = torch.bfloat16

    if config.DEMO_MODE:
        tiny_id = config.TINY_DEMO_MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(tiny_id, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            tiny_id,
            trust_remote_code=trust_remote_code,
        )
        model.to(device)
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=True,
    )

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    _ensure_rope_scaling(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=cfg,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        torch_dtype=bnb_4bit_compute_dtype,
        attn_implementation="eager",
    )

    model = prepare_model_for_kbit_training(model)

    return model, tokenizer
