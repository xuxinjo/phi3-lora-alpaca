"""Load Phi-3 Mini in 4-bit quantized mode, ready for LoRA fine-tuning."""

from typing import Optional

import torch
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PHI3_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


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

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        torch_dtype=bnb_4bit_compute_dtype,
    )

    model = prepare_model_for_kbit_training(model)

    return model, tokenizer
