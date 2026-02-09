"""Load and preprocess the Alpaca 52K dataset for instruction tuning."""

from typing import Optional

from datasets import load_dataset
from transformers import PreTrainedTokenizer


ALPACA_DATASET_ID = "tatsu-lab/alpaca"


def load_alpaca_dataset(
    split: str = "train",
    streaming: bool = False,
    trust_remote_code: bool = True,
) -> "datasets.Dataset":
    """
    Load the Alpaca 52K instruction dataset from HuggingFace.

    Args:
        split: Dataset split ('train', 'test', or 'train[:n%]' for subset).
        streaming: Whether to use streaming (for large datasets).
        trust_remote_code: Whether to trust remote code in the dataset.

    Returns:
        HuggingFace Dataset with 'instruction', 'input', 'output' columns.
    """
    dataset = load_dataset(
        ALPACA_DATASET_ID,
        split=split,
        streaming=streaming,
        trust_remote_code=trust_remote_code,
    )
    return dataset


def format_alpaca_prompt(example: dict, prompt_template: Optional[str] = None) -> str:
    """
    Format a single Alpaca example into an instruction-following prompt.

    Default format matches the original Alpaca structure:
    Below is an instruction that describes a task...
    """
    if prompt_template is None:
        prompt_template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        )

    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text.strip():
        text = prompt_template.format(
            instruction=instruction,
            input=input_text,
            output=output,
        )
    else:
        text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Response:\n{output}"
        ).format(instruction=instruction, output=output)

    return text


def prepare_dataset_for_training(
    tokenizer: PreTrainedTokenizer,
    dataset: "datasets.Dataset",
    max_length: int = 2048,
    prompt_template: Optional[str] = None,
    text_column: Optional[str] = None,
) -> "datasets.Dataset":
    """
    Tokenize and prepare the Alpaca dataset for causal LM training.

    Args:
        tokenizer: HuggingFace tokenizer (e.g. for Phi-3).
        dataset: Raw Alpaca dataset.
        max_length: Maximum sequence length (truncation).
        prompt_template: Optional custom prompt template.
        text_column: If set, use this column as pre-formatted text; else format from instruction/input/output.

    Returns:
        Dataset with 'input_ids', 'attention_mask', 'labels' ready for Trainer.
    """
    def tokenize_function(examples: dict) -> dict:
        if text_column and text_column in examples:
            texts = examples[text_column]
        else:
            texts = [
                format_alpaca_prompt(
                    {"instruction": i, "input": inp, "output": o},
                    prompt_template,
                )
                for i, inp, o in zip(
                    examples["instruction"],
                    examples["input"],
                    examples["output"],
                )
            ]

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]

        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    return tokenized_dataset
