"""Training loop for LoRA fine-tuning using HuggingFace Trainer."""

from pathlib import Path
from typing import Any, Optional

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


class LoRATrainer:
    """Orchestrates LoRA fine-tuning with HuggingFace Trainer."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        output_dir: str = "outputs",
        num_epochs: int = 3,
        per_device_train_batch_size: int = 2,
        per_device_eval_batch_size: Optional[int] = None,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.03,
        logging_steps: int = 10,
        save_strategy: str = "steps",
        save_steps: int = 100,
        save_total_limit: Optional[int] = 2,
        bf16: bool = True,
        report_to: str = "none",
        **training_kwargs: Any,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if per_device_eval_batch_size is None:
            per_device_eval_batch_size = per_device_train_batch_size

        self.training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit or 2,
            bf16=bf16,
            report_to=report_to,
            remove_unused_columns=False,
            **training_kwargs,
        )

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ) -> "Trainer":
        """
        Run training.

        Returns:
            HuggingFace Trainer after training (model state updated in place).
        """
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
        )

        trainer.train()
        trainer.save_model(str(self.output_dir / "final"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final"))

        return trainer
