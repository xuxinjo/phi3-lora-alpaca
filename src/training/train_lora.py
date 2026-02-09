"""
Train Phi-3 Mini with LoRA on Alpaca (yahma/alpaca-cleaned) using transformers.Trainer.

Loads 4-bit model + tokenizer, applies LoRA, loads dataset, trains 3 epochs,
saves adapter to checkpoints/lora_phi3.
"""

from pathlib import Path

from peft import get_peft_model
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from src.data import get_alpaca_dataloaders
from src.models import load_phi3_4bit, get_phi3_lora_config


OUTPUT_DIR = "checkpoints/lora_phi3"


def train() -> None:
    print("Loading Phi-3 Mini (4-bit)...")
    model, tokenizer = load_phi3_4bit()

    print("Applying LoRA adapters...")
    lora_config = get_phi3_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading Alpaca dataset...")
    train_loader, val_loader = get_alpaca_dataloaders(
        tokenizer=tokenizer,
        max_length=512,
        train_batch_size=64,
        val_batch_size=64,
        val_ratio=0.02,
        seed=42,
    )
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,
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
