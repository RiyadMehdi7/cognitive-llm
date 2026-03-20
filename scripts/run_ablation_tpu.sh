#!/bin/bash
# run_ablation_tpu.sh — Run the full Phase 2 ablation sweep on TPU
#
# Usage:
#   bash scripts/run_ablation_tpu.sh                     # all experiments, 3 seeds each
#   bash scripts/run_ablation_tpu.sh 01_baseline          # single experiment
#   bash scripts/run_ablation_tpu.sh --seeds 1            # quick single-seed sweep
#   bash scripts/run_ablation_tpu.sh --eval               # include lm-eval benchmarks
#
# Each experiment runs 3 seeds (42, 137, 2024) for statistical significance.
# Results are appended to results_phase2.tsv.

set -e

SEEDS="42 137 2024"
EVAL_FLAG=""
SINGLE_EXP=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            if [[ "$2" == "1" ]]; then
                SEEDS="42"
            fi
            shift 2
            ;;
        --eval)
            EVAL_FLAG="--eval_benchmarks"
            shift
            ;;
        *)
            SINGLE_EXP="$1"
            shift
            ;;
    esac
done

RESULTS_FILE="results_phase2.tsv"
LOG_DIR="experiments/phase2"
mkdir -p "$LOG_DIR"

# Write header if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo -e "exp_id\tseed\tconfig\tval_loss\tgsm8k_acc\ttimestamp" > "$RESULTS_FILE"
fi

# Get list of ablation configs
if [ -n "$SINGLE_EXP" ]; then
    CONFIGS="configs/ablation/${SINGLE_EXP}.yaml"
    if [ ! -f "$CONFIGS" ]; then
        echo "Config not found: $CONFIGS"
        echo "Available configs:"
        ls configs/ablation/*.yaml | grep -v base_tpu
        exit 1
    fi
    CONFIGS_LIST="$CONFIGS"
else
    CONFIGS_LIST=$(ls configs/ablation/[0-9]*.yaml | sort)
fi

echo "=== Cognitive LLM Phase 2 Ablation Sweep ==="
echo "Hardware: TPU (TRC)"
echo "Model: allenai/Olmo-3-1025-7B"
echo "Seeds: $SEEDS"
echo "Benchmark eval: ${EVAL_FLAG:-disabled}"
echo "Configs:"
echo "$CONFIGS_LIST" | while read cfg; do
    echo "  - $(basename "$cfg" .yaml)"
done
echo ""

TOTAL_RUNS=0
COMPLETED_RUNS=0
FAILED_RUNS=0

for config_file in $CONFIGS_LIST; do
    exp_name=$(basename "$config_file" .yaml)

    for seed in $SEEDS; do
        run_id="${exp_name}_s${seed}"
        log_file="${LOG_DIR}/${run_id}.log"

        echo "--- Running: $run_id ---"
        echo "  Config: $config_file"
        echo "  Seed: $seed"
        echo "  Log: $log_file"

        TOTAL_RUNS=$((TOTAL_RUNS + 1))

        # Run training
        if python train.py \
            --config "$config_file" \
            --seed "$seed" \
            --run_name "$run_id" \
            $EVAL_FLAG \
            2>&1 | tee "$log_file"; then

            # Extract results
            val_loss=$(grep "^val_loss:" "$log_file" | tail -1 | awk '{print $2}')
            gsm8k_acc=$(grep "^gsm8k_acc:" "$log_file" | tail -1 | awk '{print $2}' || echo "n/a")
            timestamp=$(date '+%Y-%m-%d %H:%M:%S')

            echo -e "${run_id}\t${seed}\t${exp_name}\t${val_loss}\t${gsm8k_acc}\t${timestamp}" >> "$RESULTS_FILE"

            echo "  Result: val_loss=$val_loss gsm8k_acc=$gsm8k_acc"
            COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
        else
            echo "  FAILED (see $log_file)"
            FAILED_RUNS=$((FAILED_RUNS + 1))
        fi
        echo ""
    done
done

echo "=== Ablation Sweep Complete ==="
echo "Total: $TOTAL_RUNS  Completed: $COMPLETED_RUNS  Failed: $FAILED_RUNS"
echo "Results: $RESULTS_FILE"
echo ""
echo "=== Summary ==="
if [ -f "$RESULTS_FILE" ]; then
    # Print mean val_loss per config
    echo "Config                  | Mean val_loss | Seeds"
    echo "------------------------|---------------|------"
    tail -n +2 "$RESULTS_FILE" | awk -F'\t' '{
        config[$3] = config[$3] + 0
        sum[$3] += $4
        count[$3]++
    } END {
        for (c in sum) {
            printf "%-24s| %-14.6f| %d\n", c, sum[c]/count[c], count[c]
        }
    }' | sort
fi
