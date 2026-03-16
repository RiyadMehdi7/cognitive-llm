#!/bin/bash
# run_remote.sh — SSH into remote GPU and run experiments
#
# Usage:
#   bash scripts/run_remote.sh <ssh_connection>
#
# Examples:
#   bash scripts/run_remote.sh root@208.x.x.x -p 22022     # Vast.ai
#   bash scripts/run_remote.sh user@gpu-server                # Custom server
#
# This script will:
#   1. SSH into the remote machine
#   2. Clone/update the repo
#   3. Install dependencies
#   4. Run the full experiment sweep
#   5. Copy results back to your local machine

set -e

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/run_remote.sh <ssh_args>"
    echo "Example: bash scripts/run_remote.sh root@208.x.x.x -p 22022"
    exit 1
fi

SSH_ARGS="$@"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Pushing latest code to GitHub ==="
cd "$LOCAL_DIR"
git push origin master 2>/dev/null || echo "(already up to date)"

echo ""
echo "=== Connecting to remote GPU ==="
echo "SSH args: $SSH_ARGS"

# Run setup + experiments on remote
ssh $SSH_ARGS 'bash -s' << 'REMOTE_SCRIPT'
set -e

# Setup
if [ ! -d "cognitive-llm" ]; then
    git clone https://github.com/RiyadMehdi7/cognitive-llm.git
fi
cd cognitive-llm
git pull origin master

# Install deps (quiet)
pip install -q -r requirements.txt 2>/dev/null

# Verify
python -c "import torch; assert torch.cuda.is_available(); print(f'GPU: {torch.cuda.get_device_name(0)}')"
python -c "import train; print('train.py OK')"

# Run full experiment sweep
echo ""
echo "=== Starting experiment sweep ==="
python agent.py --run 2>&1 | tee experiments/session.log

echo ""
echo "=== Done. Results: ==="
cat results.tsv
REMOTE_SCRIPT

echo ""
echo "=== Copying results back ==="
scp $SSH_ARGS:cognitive-llm/results.tsv "$LOCAL_DIR/results.tsv" 2>/dev/null || \
    echo "(scp failed — fetch results manually)"
scp -r $SSH_ARGS:cognitive-llm/experiments/ "$LOCAL_DIR/experiments/" 2>/dev/null || \
    echo "(scp experiments failed)"
scp $SSH_ARGS:cognitive-llm/lab_notebook.md "$LOCAL_DIR/lab_notebook.md" 2>/dev/null || \
    echo "(scp lab_notebook failed)"

echo ""
echo "=== Results saved locally ==="
echo "  results.tsv"
echo "  experiments/"
echo "  lab_notebook.md"
