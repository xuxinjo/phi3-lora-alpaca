"""
Load Phi-3 Mini + LoRA adapters and generate responses from instructions.

Provides generate_response(instruction) and an interactive __main__ block.
"""

from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PHI3_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_ADAPTER_PATH = "checkpoints/lora_phi3"

_model = None
_tokenizer = None


def _load_model(adapter_path: str = DEFAULT_ADAPTER_PATH):
    """Load Phi-3 Mini + LoRA adapters (lazy, once)."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter path not found: {adapter_path}. "
            "Train first with: python -m src.training.train_lora"
        )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    _tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        PHI3_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    _model = PeftModel.from_pretrained(base, adapter_path)
    _model.eval()
    return _model, _tokenizer


def _prompt(instruction: str, input_text: str = "") -> str:
    """Format like training (Instruction / Input / Response)."""
    if (input_text or "").strip():
        return f"Instruction: {instruction.strip()}\nInput: {input_text.strip()}\nResponse: "
    return f"Instruction: {instruction.strip()}\nInput: \nResponse: "


def generate_response(
    instruction: str,
    input_text: str = "",
    adapter_path: str = DEFAULT_ADAPTER_PATH,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Tokenize the instruction, generate with sampling, return the model's response.

    Args:
        instruction: User instruction.
        input_text: Optional input context (default empty).
        adapter_path: Path to LoRA adapter directory.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature (default 0.7).
        top_p: Nucleus sampling top_p (default 0.9).

    Returns:
        Generated response text.
    """
    model, tokenizer = _load_model(adapter_path)
    prompt = _prompt(instruction, input_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Response:" in full:
        response = full.split("Response:")[-1].strip()
    else:
        response = full.strip()
    return response


def run_chat(adapter_path: str = DEFAULT_ADAPTER_PATH) -> None:
    """Interactive loop: ask for instructions and print model responses."""
    print("Phi-3 Mini + LoRA instruction response")
    print("Loading model (this may take a moment)...")
    _load_model(adapter_path)
    print("Ready. Enter an instruction (or 'quit' to exit).\n")

    while True:
        try:
            instruction = input("Instruction: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not instruction:
            continue
        if instruction.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        answer = generate_response(instruction, adapter_path=adapter_path)
        print("Response:", answer)
        print()


if __name__ == "__main__":
    run_chat()
