#!/bin/bash
# Phase 2: Absolute-entropy hint resampling experiment
# Usage: bash scripts/run_phase2_resample.sh

# ===================== tunable parameters =====================
MODEL_PATH="/gpt/work/Ryan/xz/Qwen3-4B"
INPUT_JSON="/gpt/work/Ryan/xz/Qwen3-4B/phase1_zero_acc_with_entropy/zero_acc_with_entropy.json"
OUTPUT_JSON="/gpt/work/Ryan/xz/Qwen3-4B/phase2_resample_metrics.jsonl"

ALPHA=1.5          # absolute entropy threshold (0 ~ 11.9)
BETA=0.5           # fraction of qualifying positions
INSERT_HINT=true   # insert reflection hint?
HINT_TEXT="Wait, I need to pause and carefully re-examine my reasoning above. Something might be off — let me go back through each step.\n\n"
MAX_NEW_TOKENS=4096
TEMPERATURE=0.8
TOP_P=0.95

export CUDA_VISIBLE_DEVICES=1
# ==============================================================

echo "Phase 2: absolute-entropy hint resampling"
echo "  Model:      ${MODEL_PATH}"
echo "  Input:      ${INPUT_JSON}"
echo "  Output:     ${OUTPUT_JSON}"
echo "  Alpha:      ${ALPHA}"
echo "  Beta:       ${BETA}"
echo "  Insert hint: ${INSERT_HINT}"
echo "========================================"

INSERT_FLAG="--insert_hint"
if [ "${INSERT_HINT}" = "false" ]; then
    INSERT_FLAG="--no_insert_hint"
fi

python scripts/run_phase2_resample.py \
    --model_path "${MODEL_PATH}" \
    --input_json "${INPUT_JSON}" \
    --output_json "${OUTPUT_JSON}" \
    --alpha "${ALPHA}" \
    --beta "${BETA}" \
    ${INSERT_FLAG} \
    --hint_text "${HINT_TEXT}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}"

if [ $? -eq 0 ]; then
    echo "========================================"
    echo "Phase 2 complete. Results: ${OUTPUT_JSON}"
else
    echo "========================================"
    echo "Phase 2 failed."
    exit 1
fi
