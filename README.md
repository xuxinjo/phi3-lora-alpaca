# Fine-Tuning Phi-3 Mini with LoRA on the Alpaca 52K Instruction Dataset

## 1. Project Overview

This project fine-tunes **Microsoft Phi-3 Mini** using **Low-Rank Adaptation (LoRA)** on the **Alpaca 52K instruction dataset**. It provides a complete pipeline for training, evaluation, and interactive inference. The implementation uses 4-bit quantization to reduce memory requirements, making it feasible to fine-tune the model on consumer-grade GPUs.

---

## 2. Model Description (Phi-3 Mini)

**Phi-3 Mini** is a 3.8B-parameter language model from Microsoft, designed for instruction-following and conversational AI. This project uses `microsoft/Phi-3-mini-4k-instruct`, which:

- Has a 4K context window
- Is instruction-tuned out of the box
- Supports efficient 4-bit quantization via BitsAndBytes
- Works well with LoRA for parameter-efficient fine-tuning

---

## 3. Dataset Description (Alpaca 52K)

The **Alpaca 52K** dataset (`yahma/alpaca-cleaned`) is a cleaned version of the original Stanford Alpaca instruction dataset. It contains approximately 52,000 instruction–input–output triplets in the format:

| Field       | Description                          |
|------------|--------------------------------------|
| `instruction` | The task or question to perform    |
| `input`       | Optional context or input          |
| `output`      | The desired response               |

Each example is formatted as:
```
Instruction: <instruction>
Input: <input (optional)>
Response: <output>
```

---

## 4. Fine-Tuning Method (LoRA)

**Low-Rank Adaptation (LoRA)** is a parameter-efficient fine-tuning technique that freezes the base model and adds trainable low-rank matrices to specific layers. This project:

- Targets the attention projection layers: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Uses rank `r=16`, alpha `32`, and dropout `0.05`
- Trains only a small fraction of parameters (~0.1–0.5%), reducing memory and compute
- Saves only the LoRA adapter weights, which can be merged with the base model for inference

---

## 5. Project Structure

```
phi3-lora-alpaca/
├── main.py                 # CLI entry point (train, eval, chat)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── alpaca_dataset.py   # Alpaca 52K loading & formatting
    │   └── loader.py           # DataLoader utilities
    ├── evaluation/
    │   ├── __init__.py
    │   ├── evaluate.py         # Main evaluation script
    │   └── evaluator.py        # Evaluation utilities
    ├── inference/
    │   ├── __init__.py
    │   ├── generate.py         # Chat & generation logic
    │   └── predictor.py        # Inference utilities
    ├── models/
    │   ├── __init__.py
    │   ├── load_model.py       # 4-bit model loading
    │   ├── lora_config.py      # LoRA configuration
    │   ├── merge_lora.py       # Merge LoRA adapter into base model
    │   └── phi3_lora.py        # Phi-3 + LoRA setup
    └── training/
        ├── __init__.py
        ├── train_lora.py       # LoRA training loop
        └── trainer.py          # Training utilities
```

---

## 6. Installation Instructions

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Git

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/xuxinjo/phi3-lora-alpaca
   cd phi3-lora-alpaca
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Windows
   # source venv/bin/activate     # Linux/macOS
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional:** Log in to Hugging Face for model access
   ```bash
   huggingface-cli login
   ```
   *(Required if the model is gated or for private repos.)*

---

## 7. How to Run Training

Training fine-tunes Phi-3 Mini with LoRA on the Alpaca 52K dataset (3 epochs, 4-bit quantization). The LoRA adapter is saved to `checkpoints/lora_phi3` by default.

```bash
python main.py train
```

**Options:**

| Option          | Default                 | Description                          |
|-----------------|-------------------------|--------------------------------------|
| `--adapter_path`| `checkpoints/lora_phi3` | Directory to save the LoRA adapter   |

Example with a custom adapter path:
```bash
python main.py train --adapter_path my_adapters/phi3_alpaca
```

---

## 8. How to Run Evaluation

Evaluation compares the base Phi-3 Mini with the fine-tuned LoRA model on a held-out validation set. Metrics include **Perplexity**, **ROUGE-L**, and **BLEU**.

```bash
python main.py eval
```

**Options:**

| Option          | Default                 | Description                        |
|-----------------|-------------------------|------------------------------------|
| `--adapter_path`| `checkpoints/lora_phi3` | Path to the saved LoRA adapter     |
| `--seed`        | `42`                    | Random seed for validation split   |

Example:
```bash
python main.py eval --adapter_path checkpoints/lora_phi3 --seed 42
```

> **Note:** You must run training first. If the adapter path does not exist, evaluation will fail.

---

## 9. How to Run Inference (Chat Mode)

Chat mode loads the fine-tuned model and runs an interactive loop where you enter instructions and receive model responses.

```bash
python main.py chat
```

**Options:**

| Option          | Default                 | Description                        |
|-----------------|-------------------------|------------------------------------|
| `--adapter_path`| `checkpoints/lora_phi3` | Path to the saved LoRA adapter     |

Example:
```bash
python main.py chat --adapter_path checkpoints/lora_phi3
```

Type `quit`, `exit`, or `q` to exit the chat.

---

## 10. Example Commands

```bash
# Train the model (3 epochs on Alpaca 52K)
python main.py train

# Evaluate base vs fine-tuned model
python main.py eval

# Interactive chat with the fine-tuned model
python main.py chat
```

With custom adapter path:
```bash
python main.py train --adapter_path checkpoints/lora_phi3
python main.py eval --adapter_path checkpoints/lora_phi3
python main.py chat --adapter_path checkpoints/lora_phi3
```

---

## 11. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU      | 8 GB VRAM (e.g., RTX 3070) | 16+ GB VRAM (e.g., RTX 4080, A100) |
| RAM      | 16 GB                      | 32 GB                               |
| Storage  | ~10 GB (model + dataset)   | 20+ GB (checkpoints, logs)          |
| CUDA     | 11.8+                      | 12.x                                |

With 4-bit quantization, training is feasible on GPUs with 8–12 GB VRAM. Larger batch sizes or longer sequences may require more memory.

---

## 12. Notes on 4-bit Quantization

- **BitsAndBytes** is used to load Phi-3 Mini in **4-bit NF4** format, reducing memory by roughly 4× compared to full precision.
- **Double quantization** is enabled to further reduce memory overhead.
- Compute is performed in **bfloat16** for numerical stability.
- LoRA adapters are trained on top of the quantized model; only the adapter weights are stored and can be applied to the base model at inference time.
- Quantization may slightly affect generation quality but allows fine-tuning on consumer hardware.

---

## 13. License

This project is provided for research and educational purposes. Usage is subject to:

- **Microsoft Phi-3** terms: [Phi-3 License](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- **Alpaca dataset** terms: [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)

Please review the respective licenses before commercial use.

---

## 14. Acknowledgements

- **[Hugging Face](https://huggingface.co/)** — Transformers, PEFT, Datasets, and model hosting
- **[Microsoft](https://www.microsoft.com/)** — Phi-3 Mini language model
- **[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)** — Original Alpaca instruction dataset
- **yahma** — [alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset
