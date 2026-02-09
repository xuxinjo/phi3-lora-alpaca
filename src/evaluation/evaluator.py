"""Evaluation metrics for the fine-tuned model."""

from typing import Optional

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset,
    output_dir: str = "eval_output",
    per_device_eval_batch_size: int = 4,
) -> dict[str, float]:
    """
    Run evaluation and return loss and perplexity.

    Args:
        model: Fine-tuned causal LM.
        tokenizer: Tokenizer.
        eval_dataset: Dataset with 'input_ids', 'attention_mask', 'labels'.
        output_dir: Directory for eval logs.
        per_device_eval_batch_size: Eval batch size.

    Returns:
        Dict with 'eval_loss' and 'eval_perplexity'.
    """
    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=per_device_eval_batch_size,
        do_train=False,
        do_eval=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    metrics = trainer.evaluate()
    if "eval_loss" in metrics:
        import math
        metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])

    return metrics
