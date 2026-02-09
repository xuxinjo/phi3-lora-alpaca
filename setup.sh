#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment $VENV_DIR already exists."
fi

source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip --quiet

if [[ -f "$REQUIREMENTS" ]]; then
    echo "Installing dependencies from $REQUIREMENTS..."
    pip install -r "$REQUIREMENTS" --quiet
else
    echo "Error: $REQUIREMENTS not found." >&2
    exit 1
fi

echo "Verifying installation..."
python3 -c "
packages = ['torch', 'transformers', 'datasets', 'peft', 'bitsandbytes', 'accelerate', 'evaluate']
failed = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'  {pkg}: OK')
    except ImportError:
        failed.append(pkg)
        print(f'  {pkg}: FAILED')
if failed:
    raise SystemExit(f'Missing packages: {failed}')
"

echo ""
echo "Environment setup complete."
