#!/bin/bash
# DERPO 消融实验 — 1.7B, ~50q, 400 steps
#   (1) DERPO full:   ΔH trigger + hint
#   (2) w/o ΔH trigger: absolute H trigger
#   (3) w/o hint v:   ΔH trigger, no hint
set -e

export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_BIN="${SCRIPT_DIR}/openr1/bin"
export PATH="${VENV_BIN}:${PATH}"
TMP_DIR=$(mktemp -d -t ablation_XXXX)
trap "rm -rf $TMP_DIR" EXIT

MODEL_PATH="/gpt/work/Ryan/xz/Qwen3-1.7B"
INPUT_JSON="/gpt/work/Ryan/xz/Qwen3-4B/phase1_zero_acc_with_entropy_idea6/zero_acc_with_entropy.json"
BASE_DIR="/gpt/work/Ryan/xz/Qwen3-1.7B/ablation"
ACCEL_CONFIG="recipes/accelerate_configs/zero2.yaml"

# 实验定义: "名称|alpha|beta|insert_hint|use_absolute_h"
EXPERIMENTS=(
  "DERPO_full|0.5|0.0|true|false"
  "wo_deltaH|0.5|0.0|true|true"
  "wo_hint|0.5|0.0|false|false"
)

_gen_yaml() {
  local output_dir="$1" alpha="$2" beta="$3" insert_hint="$4" use_abs="$5"
  local metrics_json="${output_dir}/phase2_metrics.jsonl"
  python3 - "$output_dir" "$alpha" "$beta" "$insert_hint" "$use_abs" "$metrics_json" "$MODEL_PATH" "$INPUT_JSON" <<'PYEOF'
import sys, yaml
output_dir = sys.argv[1]
alpha = float(sys.argv[2])
beta = float(sys.argv[3])
insert_hint = sys.argv[4].lower() == "true"
use_abs = sys.argv[5].lower() == "true"
metrics_json = sys.argv[6]
model_path = sys.argv[7]
input_json = sys.argv[8]

config = {
    "model_name_or_path": model_path,
    "model_revision": "main",
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
    "dataset_name": "phase2_json",
    "phase2_input_json": input_json,
    "phase2_output_json": metrics_json,
    "phase2_alpha": alpha,
    "phase2_beta": beta,
    "phase2_insert_hint": insert_hint,
    "phase2_hint_text": "Wait, I need to pause and carefully re-examine my reasoning above. Something might be off — let me go back through each step.\n\n",
    "phase2_skip_samples": 0,
    "phase2_use_absolute_h": use_abs,
    "system_prompt": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed{} . The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.",
    "bf16": True,
    "use_vllm": True,
    "vllm_gpu_memory_utilization": 0.35,
    "vllm_max_model_len": 8192,
    "beta": 0.01,
    "temperature": 0.8,
    "do_eval": False,
    "gradient_accumulation_steps": 8,
    "gradient_checkpointing": True,
    "use_liger_loss": False,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "learning_rate": 2.0e-06,
    "log_completions": True,
    "logging_first_step": True,
    "logging_steps": 1,
    "lr_scheduler_type": "cosine",
    "max_prompt_length": 512,
    "max_completion_length": 6120,
    "max_steps": 400,
    "num_train_epochs": -1,
    "num_generations": 4,
    "output_dir": output_dir,
    "overwrite_output_dir": False,
    "per_device_eval_batch_size": 1,
    "per_device_train_batch_size": 1,
    "push_to_hub": False,
    "reward_funcs": ["cosine", "tag_count"],
    "reward_weights": [1.0, 1.0],
    "cosine_min_value_wrong": 0.0,
    "cosine_max_value_wrong": -0.5,
    "cosine_min_value_correct": 0.5,
    "cosine_max_value_correct": 1.0,
    "cosine_max_len": 40000,
    "save_strategy": "steps",
    "save_steps": 400,
    "wandb_log_unique_prompts": False,
    "save_total_limit": 1,
    "seed": 50,
    "shuffle_dataset": False,
    "warmup_ratio": 0.1,
    "use_peft": False,
    "report_to": [],
}
print(yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False))
PYEOF
}

TOTAL=${#EXPERIMENTS[@]}
CURRENT=0

for entry in "${EXPERIMENTS[@]}"; do
  CURRENT=$((CURRENT + 1))
  IFS='|' read -r NAME ALPHA BETA INSERT_HINT USE_ABS <<< "$entry"
  OUTPUT_DIR="${BASE_DIR}/${NAME}"

  echo ""
  echo "============================================================"
  echo "[${CURRENT}/${TOTAL}] ${NAME}"
  echo "   alpha=${ALPHA} beta=${BETA} hint=${INSERT_HINT} absH=${USE_ABS}"
  echo "   输出: ${OUTPUT_DIR}"
  echo "============================================================"

  if [ -d "${OUTPUT_DIR}" ] && ls "${OUTPUT_DIR}"/checkpoint-*/trainer_state.json >/dev/null 2>&1; then
    echo "[${CURRENT}/${TOTAL}] SKIP  已有 checkpoint"
    continue
  fi

  rm -rf "${OUTPUT_DIR}"
  YAML_FILE="${TMP_DIR}/config_${NAME}.yaml"
  _gen_yaml "${OUTPUT_DIR}" "${ALPHA}" "${BETA}" "${INSERT_HINT}" "${USE_ABS}" > "$YAML_FILE"
  echo "   配置: ${YAML_FILE}"

  echo "[${CURRENT}/${TOTAL}] 开始训练..."
  cd "${SCRIPT_DIR}"

  set +e
  accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes=1 \
    src/open_r1/grpo_phase2.py \
    --config "${YAML_FILE}" \
    --vllm_mode colocate
  RC=$?
  set -e

  if [ $RC -ne 0 ]; then
    if [ -d "${OUTPUT_DIR}" ] && ls "${OUTPUT_DIR}"/checkpoint-*/trainer_state.json >/dev/null 2>&1; then
      echo "[${CURRENT}/${TOTAL}] WARNING: exit=${RC}，checkpoint 已保存，继续"
    else
      echo "[${CURRENT}/${TOTAL}] ERROR: exit=${RC}，无 checkpoint"
    fi
  else
    echo "[${CURRENT}/${TOTAL}] DONE → ${OUTPUT_DIR}"
  fi
done

echo ""
echo "============================================================"
echo "  消融实验全部完成！共 ${TOTAL} 个实验"
echo "============================================================"
