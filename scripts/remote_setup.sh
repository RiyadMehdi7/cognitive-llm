#!/bin/bash
# remote_setup.sh — Run this ONCE on the remote GPU machine
# Usage: bash scripts/remote_setup.sh

set -e

echo "=== Cognitive LLM Remote Setup ==="

# Clone repo
if [ ! -d "cognitive-llm" ]; then
    git clone https://github.com/RiyadMehdi7/cognitive-llm.git
fi
cd cognitive-llm

# Pull latest
git pull origin master

# Install dependencies
pip install -q -r requirements.txt

# Verify GPU
python -c "
import torch
assert torch.cuda.is_available(), 'No CUDA GPU found!'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# Verify imports
python -c "import train; print('train.py OK')"
python agent.py

echo ""
echo "=== Setup complete. Run experiments with: ==="
echo "  cd cognitive-llm && python agent.py --run"
