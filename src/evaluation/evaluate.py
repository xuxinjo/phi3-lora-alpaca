"""
Evaluate fine-tuned LoRA Phi-3 vs base Phi-3 on a held-out validation set.

- Loads base and fine-tuned models
- Computes perplexity on validation set
- Computes ROUGE-L and BLEU on 200 random samples
- Prints before/after comparison
"""

import argparse
import math
import random
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src import config
from src.models import load_phi3_4bit


PHI3_MODEL_ID = config.PHI3_MODEL_ID
ALPACA_DATASET_ID = "yahma/alpaca-cleaned"
DEFAULT_ADAPTER_PATH = "checkpoints/lora_phi3"
NUM_ROUGE_BLEU_SAMPLES = 200
PERPLEXITY_MAX_SAMPLES = 500
MAX_LENGTH = 512
MAX_NEW_TOKENS = 256


def _prompt_for_generation(instruction: str, input_text: str) -> str:
    """Same format as training (alpaca_dataset)."""
    if (input_text or "").strip():
        return f"Instruction: {instruction.strip()}\nInput: {input_text.strip()}\nResponse: "
    return f"Instruction: {instruction.strip()}\nInput: \nResponse: "


def _format_full_text(instruction: str, input_text: str, output: str) -> str:
    """Full sequence for perplexity (instruction + input + response)."""
    return _prompt_for_generation(instruction, input_text or "") + (output or "").strip()


def load_base_model():
    """Load base model (Phi-3 or tiny demo model)."""
    model, tokenizer = load_phi3_4bit()
    return model, tokenizer


def load_finetuned_model(adapter_path: str):
    """Load base model and apply LoRA adapter; tokenizer from base model."""
    model, tokenizer = load_phi3_4bit()
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[dict],
    max_length: int = MAX_LENGTH,
    batch_size: int = 2,
) -> float:
    """Compute mean perplexity (exp(mean loss)) on examples."""
    model.eval()
    total_loss = 0.0
    n = 0
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        texts = [
            _format_full_text(
                ex["instruction"],
                ex.get("input") or "",
                ex["output"],
            )
            for ex in batch
        ]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        enc["labels"] = enc["input_ids"].clone()
        enc["labels"][enc["attention_mask"] == 0] = -100

        with torch.no_grad():
            out = model(**enc)
            loss = out.loss
        total_loss += loss.item() * len(batch)
        n += len(batch)

    return math.exp(total_loss / n) if n else float("nan")


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[dict],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[str]:
    """Generate response for each example (instruction + input -> model output)."""
    model.eval()
    generated = []
    for ex in tqdm(examples, desc="Generating"):
        prompt = _prompt_for_generation(
            ex["instruction"],
            ex.get("input") or "",
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        full = tokenizer.decode(out[0], skip_special_tokens=True)
        if "Response:" in full:
            response = full.split("Response:")[-1].strip()
        else:
            response = full.strip()
        generated.append(response)
    return generated


def compute_rouge_bleu(predictions: list[str], references: list[str]) -> dict:
    """Return dict with rougeL and bleu scores."""
    try:
        evaluate = __import__("evaluate")
    except ImportError:
        raise ImportError("Install 'evaluate' for ROUGE/BLEU: pip install evaluate")

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    rouge_res = rouge.compute(predictions=predictions, references=references)
    bleu_res = bleu.compute(predictions=predictions, references=[[r] for r in references])

    return {
        "rougeL": rouge_res.get("rougeL", rouge_res.get("rouge_l", 0.0)),
        "bleu": bleu_res.get("bleu", 0.0),
    }


def run_evaluation(adapter_path: str, seed: int = 42, demo: bool = False) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if demo:
        config.DEMO_MODE = True
        print("Running evaluation in DEMO MODE")
        model_id = config.TINY_DEMO_MODEL_ID

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

        # Tiny synthetic evaluation set.
        examples = [
            {
                "instruction": "Say hello.",
                "input": "",
                "output": "Hello from demo mode.",
            },
            {
                "instruction": "Explain what this demo is doing.",
                "input": "",
                "output": "It runs a lightweight training and evaluation loop.",
            },
        ]

        perp_examples = examples

        print("\n--- Demo base model (tiny GPT-style model) ---")
        base_ppl = compute_perplexity(base_model, tokenizer, perp_examples, max_length=64, batch_size=1)
        print(f"  Perplexity (demo): {base_ppl:.4f}")

        # For demo mode we keep dependencies minimal: skip ROUGE/BLEU, just
        # show a couple of generated responses and perplexity.
        print("\nDemo sample generations:")
        for ex in examples:
            prompt = _prompt_for_generation(ex["instruction"], ex.get("input") or "")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = base_model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            full = tokenizer.decode(out[0], skip_special_tokens=True)
            if "Response:" in full:
                pred = full.split("Response:")[-1].strip()
            else:
                pred = full.strip()
            print(f"- Instruction: {ex['instruction']}")
            print(f"  Target:      {ex['output']}")
            print(f"  Prediction:  {pred}\n")

        print("Demo evaluation complete (no LoRA, no Alpaca dataset loaded, ROUGE/BLEU skipped).")
        return

    config.DEMO_MODE = False

    print("Loading validation data (yahma/alpaca-cleaned)...")
    ds = load_dataset(ALPACA_DATASET_ID, split="train", trust_remote_code=True)
    split = ds.train_test_split(test_size=0.05, seed=seed)
    val = split["test"]
    if config.DEMO_MODE:
        max_samples = min(config.DEMO_DATASET_MAX_SAMPLES, len(val))
        val = val.select(range(max_samples))

    val_list = [val[i] for i in range(len(val))]

    if config.DEMO_MODE:
        max_perp = min(config.DEMO_DATASET_MAX_SAMPLES, len(val_list))
        perp_examples = val_list[:max_perp]
    else:
        perp_examples = val_list[: min(PERPLEXITY_MAX_SAMPLES, len(val_list))]

    if config.DEMO_MODE:
        sample_count = min(config.DEMO_DATASET_MAX_SAMPLES, len(val_list))
        rouge_bleu_indices = list(range(sample_count))
    else:
        if len(val_list) >= NUM_ROUGE_BLEU_SAMPLES:
            rouge_bleu_indices = random.sample(range(len(val_list)), NUM_ROUGE_BLEU_SAMPLES)
        else:
            rouge_bleu_indices = list(range(len(val_list)))
    rouge_bleu_examples = [val_list[i] for i in rouge_bleu_indices]
    references = [ex["output"].strip() for ex in rouge_bleu_examples]

    print("\n--- Base Phi-3 Mini (4-bit, no LoRA) ---")
    print("Loading base model...")
    base_model, tokenizer = load_base_model()

    print("Computing perplexity (base)...")
    base_ppl = compute_perplexity(base_model, tokenizer, perp_examples)
    print(f"  Perplexity: {base_ppl:.4f}")

    print("Generating 200 samples (base)...")
    base_predictions = generate_responses(base_model, tokenizer, rouge_bleu_examples)
    del base_model
    torch.cuda.empty_cache()

    base_metrics = compute_rouge_bleu(base_predictions, references)
    print(f"  ROUGE-L: {base_metrics['rougeL']:.4f}")
    print(f"  BLEU:    {base_metrics['bleu']:.4f}")

    print("\n--- Fine-tuned Phi-3 (LoRA) ---")
    print(f"Loading adapter from {adapter_path}...")
    ft_model, _ = load_finetuned_model(adapter_path)

    print("Computing perplexity (fine-tuned)...")
    ft_ppl = compute_perplexity(ft_model, tokenizer, perp_examples)
    print(f"  Perplexity: {ft_ppl:.4f}")

    print("Generating 200 samples (fine-tuned)...")
    ft_predictions = generate_responses(ft_model, tokenizer, rouge_bleu_examples)
    del ft_model
    torch.cuda.empty_cache()

    ft_metrics = compute_rouge_bleu(ft_predictions, references)
    print(f"  ROUGE-L: {ft_metrics['rougeL']:.4f}")
    print(f"  BLEU:    {ft_metrics['bleu']:.4f}")

    print("\n" + "=" * 60)
    print("BEFORE / AFTER COMPARISON (Base Phi-3 vs Fine-tuned Phi-3)")
    print("=" * 60)
    print(f"{'Metric':<20} {'Base Phi-3':>14} {'Fine-tuned':>14} {'Change':>12}")
    print("-" * 60)
    print(f"{'Perplexity (lower is better)':<20} {base_ppl:>14.4f} {ft_ppl:>14.4f} {ft_ppl - base_ppl:>+12.4f}")
    print(f"{'ROUGE-L (higher is better)':<20} {base_metrics['rougeL']:>14.4f} {ft_metrics['rougeL']:>14.4f} {ft_metrics['rougeL'] - base_metrics['rougeL']:>+12.4f}")
    print(f"{'BLEU (higher is better)':<20} {base_metrics['bleu']:>14.4f} {ft_metrics['bleu']:>14.4f} {ft_metrics['bleu'] - base_metrics['bleu']:>+12.4f}")
    print("=" * 60)

    n_examples = min(3, len(rouge_bleu_examples))
    print("\n--- Example outputs (first {} samples) ---\n".format(n_examples))
    for idx in range(n_examples):
        ex = rouge_bleu_examples[idx]
        print(f"[Sample {idx + 1}] Instruction: {ex['instruction'][:80]}...")
        print(f"  Reference:     {references[idx][:120]}...")
        print(f"  Base Phi-3:    {base_predictions[idx][:120]}...")
        print(f"  Fine-tuned:    {ft_predictions[idx][:120]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned Phi-3 LoRA")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=DEFAULT_ADAPTER_PATH,
        help="Path to saved LoRA adapter (default: checkpoints/lora_phi3)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not Path(args.adapter_path).exists():
        raise FileNotFoundError(
            f"Adapter path not found: {args.adapter_path}. "
            "Train first with: python -m src.training.train_lora"
        )

    run_evaluation(adapter_path=args.adapter_path, seed=args.seed)


if __name__ == "__main__":
    main()
