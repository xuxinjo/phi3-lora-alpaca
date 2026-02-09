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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from src import config
from src.data import get_alpaca_dataloaders
from src.models import load_phi3_4bit, get_phi3_lora_config


OUTPUT_DIR = "checkpoints/lora_phi3"


def train(demo: bool = False) -> None:
    """
    Train entry point.

    - demo=False: full Phi-3 + LoRA training on Alpaca.
    - demo=True: tiny model, tiny synthetic dataset, very fast loop.
    """
    if demo:
        config.DEMO_MODE = True
        print("Running in DEMO MODE")
        model_id = config.TINY_DEMO_MODEL_ID

        print(f"Loading tiny demo model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        model.train()

        # Tiny synthetic dataset (20 samples).
        texts = [
            f"Instruction: Demo instruction {i}\nInput: \nResponse: Demo response {i}."
            for i in range(20)
        ]
        encodings = [
            tokenizer(
                t,
                return_tensors="pt",
                max_length=config.DEMO_MAX_LENGTH,
                truncation=True,
                padding="max_length",
            )
            for t in texts
        ]

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        print("Starting demo training loop (20 steps)...")
        for step in tqdm(range(20), desc="Demo training"):
            batch = encodings[step % len(encodings)]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            time.sleep(0.05)  # make progress visible but very light

        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Demo training complete. Tiny model saved to {OUTPUT_DIR}")
        return

    # Full mode: standard Phi-3 + LoRA training.
    config.DEMO_MODE = False

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
