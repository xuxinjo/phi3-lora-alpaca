"""
CLI for Phi-3 Mini LoRA on Alpaca: train, evaluate, or chat.

Usage:
    python main.py train
    python main.py eval [--adapter_path checkpoints/lora_phi3]
    python main.py chat [--adapter_path checkpoints/lora_phi3]
"""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Phi-3 Mini LoRA on Alpaca: train, evaluate, or chat",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")

    train_parser = subparsers.add_parser("train", help="Run LoRA fine-tuning")
    train_parser.set_defaults(func=_run_train)
    train_parser.add_argument(
        "--adapter_path",
        type=str,
        default="checkpoints/lora_phi3",
        help="Directory to save LoRA adapter (default: checkpoints/lora_phi3)",
    )

    eval_parser = subparsers.add_parser("eval", help="Run evaluation (base vs fine-tuned)")
    eval_parser.set_defaults(func=_run_eval)
    eval_parser.add_argument(
        "--adapter_path",
        type=str,
        default="checkpoints/lora_phi3",
        help="Path to saved LoRA adapter (default: checkpoints/lora_phi3)",
    )
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed for val split")

    chat_parser = subparsers.add_parser("chat", help="Run inference (interactive chat)")
    chat_parser.set_defaults(func=_run_chat)
    chat_parser.add_argument(
        "--adapter_path",
        type=str,
        default="checkpoints/lora_phi3",
        help="Path to saved LoRA adapter (default: checkpoints/lora_phi3)",
    )

    args = parser.parse_args()
    args.func(args)


def _run_train(args: argparse.Namespace) -> None:
    from src.training.train_lora import train

    import src.training.train_lora as train_lora
    train_lora.OUTPUT_DIR = args.adapter_path
    train()


def _run_eval(args: argparse.Namespace) -> None:
    from src.evaluation.evaluate import run_evaluation

    run_evaluation(adapter_path=args.adapter_path, seed=args.seed)


def _run_chat(args: argparse.Namespace) -> None:
    from src.inference.generate import run_chat

    run_chat(adapter_path=args.adapter_path)


if __name__ == "__main__":
    main()
