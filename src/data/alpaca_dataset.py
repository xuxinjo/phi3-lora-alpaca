"""Load Alpaca 52K (yahma/alpaca-cleaned), preprocess, tokenize, and return DataLoaders."""

from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from src import config


ALPACA_DATASET_ID = "yahma/alpaca-cleaned"


def _format_sample(instruction: str, input_text: str, output: str) -> str:
    """Format a single example into the required text string."""
    parts = [
        f"Instruction: {instruction.strip()}",
        f"Input: {input_text.strip()}" if input_text.strip() else "Input: ",
        f"Response: {output.strip()}",
    ]
    return "\n".join(parts)


def get_alpaca_dataloaders(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    train_batch_size: int = 8,
    val_batch_size: Optional[int] = None,
    val_ratio: float = 0.02,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Load yahma/alpaca-cleaned, preprocess into Instruction/Input/Response text,
    tokenize with the given tokenizer (truncate to max_length), and return
    training and validation PyTorch DataLoaders.

    Args:
        tokenizer: Phi-3 (or compatible) tokenizer.
        max_length: Maximum sequence length (truncation).
        train_batch_size: Batch size for training DataLoader.
        val_batch_size: Batch size for validation; defaults to train_batch_size.
        val_ratio: Fraction of data used for validation (train_test_split).
        seed: Random seed for train/val split.
        num_workers: DataLoader num_workers.

    Returns:
        (train_dataloader, val_dataloader).
    """
    if config.DEMO_MODE:
        # Override settings for a very small, fast demo run.
        max_length = min(max_length, config.DEMO_MAX_LENGTH)
        train_batch_size = config.DEMO_TRAIN_BATCH_SIZE
        val_batch_size = config.DEMO_VAL_BATCH_SIZE
        val_ratio = min(val_ratio, 0.1)

    if val_batch_size is None:
        val_batch_size = train_batch_size

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(ALPACA_DATASET_ID, split="train", trust_remote_code=True)
    if config.DEMO_MODE:
        max_samples = min(config.DEMO_DATASET_MAX_SAMPLES, len(dataset))
        dataset = dataset.select(range(max_samples))
    split = dataset.train_test_split(test_size=val_ratio, seed=seed)
    train_ds = split["train"]
    val_ds = split["test"]

    train_dataset = _AlpacaTokenizedDataset(
        train_ds,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = _AlpacaTokenizedDataset(
        val_ds,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


class _AlpacaTokenizedDataset(Dataset):
    """PyTorch Dataset of tokenized Alpaca samples with labels for causal LM."""

    def __init__(
        self,
        hf_dataset: "datasets.Dataset",
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict:
        row = self.hf_dataset[idx]
        text = _format_sample(
            row["instruction"],
            row.get("input", "") or "",
            row["output"],
        )
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = [x if attention_mask[i] else -100 for i, x in enumerate(input_ids)]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _collate_fn(batch: list[dict]) -> dict:
    """Stack batch tensors (lists of lists from tokenizer)."""
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
