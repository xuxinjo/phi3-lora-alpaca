"""Inference with the fine-tuned Phi-3 LoRA model."""

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Phi3Predictor:
    """Generate responses using a fine-tuned Phi-3 + LoRA model."""

    def __init__(
        self,
        model_path: str,
        base_model_id: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
    ):
        """
        Load model and tokenizer from a checkpoint directory.

        Args:
            model_path: Path to saved LoRA adapter + tokenizer (e.g. outputs/final).
            base_model_id: Base Phi-3 model ID if loading adapter only; else model_path is full model.
            device: Device to run on ('cuda', 'cpu', or None for auto).
            torch_dtype: Dtype for model (e.g. torch.bfloat16).
            trust_remote_code: Allow custom model code.
        """
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        if base_model_id:
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if device is None else device,
                trust_remote_code=trust_remote_code,
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto" if device is None else device,
                trust_remote_code=trust_remote_code,
            )

        self.model.eval()

    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        Generate a response for the given instruction (and optional input).

        Args:
            instruction: The instruction string.
            input_text: Optional input context.
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to sample; if False, greedy decode.
            temperature: Sampling temperature.
            top_p: Nucleus sampling top_p.
            **kwargs: Passed to model.generate().

        Returns:
            Generated response text (only the model output part).
        """
        if input_text.strip():
            prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n"
            ).format(instruction=instruction, input=input_text)
        else:
            prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### Response:\n"
            ).format(instruction=instruction)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **kwargs,
            )

        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Response:" in full:
            return full.split("### Response:")[-1].strip()
        return full.strip()
