"""Load Phi-3 Mini in 4-bit quantized mode, ready for LoRA fine-tuning."""

from typing import Any, Optional

import torch
from peft import prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PHI3_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


def _ensure_rope_scaling(config: Any) -> None:
    # Phi-3 Hub config has rope_scaling: null. The Hub's modeling_phi3.py uses Phi3RotaryEmbedding
    # when rope_scaling is None (correct for 4k). If we set a dict, it only accepts type "longrope"
    # (and needs short_factor, long_factor, original_max_position_embeddings). So leave None as-is.
    current = getattr(config, "rope_scaling", None)
    if current is None:
        return  # leave None -> standard RoPE for 4k context
    if isinstance(current, dict) and current.get("type") is None:
        # Config has rope_scaling dict but missing "type"; use "longrope" for extended context.
        max_pos = getattr(config, "max_position_embeddings", 4096)
        orig_max = getattr(config, "original_max_position_embeddings", max_pos)
        config.rope_scaling = {
            **current,
            "type": "longrope",
            "short_factor": current.get("short_factor", [1.0]),
            "long_factor": current.get("long_factor", [1.0]),
            "factor": current.get("factor", 1.0),
        }
        if not hasattr(config, "original_max_position_embeddings"):
            config.original_max_position_embeddings = orig_max
    if hasattr(config, "__dict__"):
        config.__dict__["rope_scaling"] = config.rope_scaling


def load_phi3_4bit(
    model_id: str = PHI3_MODEL_ID,
    bnb_4bit_compute_dtype: Optional[torch.dtype] = None,
    bnb_4bit_quant_type: str = "nf4",
    trust_remote_code: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Phi-3 Mini with 4-bit quantization and prepare it for LoRA fine-tuning.

    - Loads the tokenizer from the model repo.
    - Loads the model with device_map="auto" and bitsandbytes 4-bit config.
    - Calls prepare_model_for_kbit_training so LoRA can be applied later.

    Args:
        model_id: HuggingFace model ID (default: Phi-3 Mini 4k instruct).
        bnb_4bit_compute_dtype: Dtype for 4-bit compute (e.g. torch.bfloat16). Defaults to bfloat16.
        bnb_4bit_quant_type: Quantization type: "nf4" or "fp4".
        trust_remote_code: Whether to trust custom model code.

    Returns:
        (model, tokenizer) with model ready for LoRA (e.g. get_peft_model).
    """
    if bnb_4bit_compute_dtype is None:
        bnb_4bit_compute_dtype = torch.bfloat16

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

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    _ensure_rope_scaling(config)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        torch_dtype=bnb_4bit_compute_dtype,
        attn_implementation="eager",
    )

    model = prepare_model_for_kbit_training(model)

    return model, tokenizer
