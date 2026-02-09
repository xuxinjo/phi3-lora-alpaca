"""
Training entry point.

- In full mode: trains Phi-3 Mini with LoRA on Alpaca using transformers.Trainer.
- In demo mode: runs a very lightweight loop on a tiny model and a small subset
  of the dataset, with a fast-moving progress bar.
"""

from pathlib import Path
import time

import torch
from peft import get_peft_model
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from src import config
from src.data import get_alpaca_dataloaders
from src.models import load_phi3_4bit, get_phi3_lora_config


OUTPUT_DIR = "checkpoints/lora_phi3"


def _demo_train_loop(model, train_loader, tokenizer) -> None:
    """Very small, fast training loop for demo mode."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()

    print("Starting demo training (tiny model, 1 epoch)...")
    steps = len(train_loader)
    progress = tqdm(enumerate(train_loader, start=1), total=steps, desc="Demo training")

    for step, batch in progress:
        time.sleep(0.05)  # keep things light but visibly active
        batch = {k: torch.tensor(v, device=device) if not isinstance(v, torch.Tensor) else v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Demo LoRA adapter and tokenizer saved to {OUTPUT_DIR}")


def train() -> None:
    if config.DEMO_MODE:
        print("Running in demo mode (tiny model, tiny dataset).")

    print("Loading model...")
    model, tokenizer = load_phi3_4bit()

    print("Applying LoRA adapters...")
    lora_config = get_phi3_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading Alpaca dataset...")
    train_loader, val_loader = get_alpaca_dataloaders(
        tokenizer=tokenizer,
        max_length=256,
        train_batch_size=8,
        val_batch_size=4,
        val_ratio=0.02,
        seed=42,
    )
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    # Demo mode: use the lightweight custom loop.
    if config.DEMO_MODE:
        _demo_train_loop(model, train_loader, tokenizer)
        return

    # Full mode: standard Trainer setup.
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Starting training (3 epochs)...")
    trainer.train()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter and tokenizer saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
