#!/bin/bash
# Intuitor 训练 — 自动化 1.7B / 4B / 8B
# Self-certainty reward, 单卡 GPU 1
set -e

export CUDA_VISIBLE_DEVICES=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_BIN="${SCRIPT_DIR}/openr1/bin"
export PATH="${VENV_BIN}:${PATH}"
TMP_DIR=$(mktemp -d -t intuitor_XXXX)
trap "rm -rf $TMP_DIR" EXIT

# 格式: "模型路径|accelerate_config|输出子目录|gpu_memory_util|max_completion_len|vllm_max_model_len"
MODELS=(
  "/gpt/work/Ryan/xz/Qwen3-1.7B|zero2.yaml|origin_Intuitor|0.35|3072|4096"
  "/gpt/work/Ryan/xz/Qwen3-4B|zero3_offload.yaml|origin_Intuitor|0.45|3072|4096"
  "/gpt/work/Ryan/xz/Qwen3-8B|zero3_offload.yaml|origin_Intuitor|0.60|3072|4096"
)

_gen_yaml() {
  local model_path="$1" output_dir="$2" gpu_util="$3" max_comp_len="$4" vllm_max_len="$5"
  python3 - "$model_path" "$output_dir" "$gpu_util" "$max_comp_len" "$vllm_max_len" <<'PYEOF'
import sys, yaml

model_path, output_dir = sys.argv[1], sys.argv[2]
gpu_util = float(sys.argv[3])
max_comp_len = int(sys.argv[4])
vllm_max_len = int(sys.argv[5])

config = {
    "model_name_or_path": model_path,
    "model_revision": "main",
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
    "dataset_mixture": {
        "datasets": [
            {"id": "GAIR/limo", "config": "default", "split": "train", "columns": ["question", "solution"], "weight": 1.0},
            {"id": "/gpt/work/Ryan/xz/open-r1/data/zero_acc_questions", "split": "train", "columns": ["question", "solution"], "weight": 1.0},
        ],
        "seed": 42,
    },
    "dataset_prompt_column": "question",
    "system_prompt": "You are a helpful AI Assistant, designed to provided well-reasoned and detailed responses. You FIRST think about the reasoning process step by step and then provide the user with the answer. Please enclose your final answer in the box: \\boxed{Your Answer}. Please stop generation immediately after outputing the box.",
    "bf16": True,
    "use_vllm": True,
    "vllm_gpu_memory_utilization": gpu_util,
    "vllm_max_model_len": vllm_max_len,
    "beta": 0.005,
    "do_eval": False,
    "gradient_accumulation_steps": 16,
    "gradient_checkpointing": True,
    "use_liger_loss": False,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "learning_rate": 3.0e-06,
    "log_completions": True,
    "logging_first_step": True,
    "logging_steps": 5,
    "lr_scheduler_type": "cosine",
    "max_prompt_length": 512,
    "max_completion_length": max_comp_len,
    "num_train_epochs": 1,
    "num_generations": 4,
    "num_iterations": 1,
    "per_device_eval_batch_size": 1,
    "per_device_train_batch_size": 1,
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "seed": 7,
    "warmup_ratio": 0.1,
    "reward_funcs": ["accuracy"],
    "reward_weights": [0.0],  # Intuitor: 只监控 accuracy，训练用 self-certainty
    "output_dir": output_dir,
    "overwrite_output_dir": False,
    "push_to_hub": False,
    "wandb_log_unique_prompts": False,
    "use_peft": False,
    "report_to": ["wandb"],
}
print(yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False))
PYEOF
}

_run_training() {
  local OUTPUT_DIR="$1"
  cd "${SCRIPT_DIR}"
  set +e
  accelerate launch \
    --config_file "recipes/accelerate_configs/${ACCEL_CONFIG}" \
    --num_processes=1 \
    src/open_r1/intuitor_grpo.py \
    --config "${YAML_FILE}" \
    --vllm_mode colocate
  local TRAIN_EXIT=$?
  set -e
  if [ $TRAIN_EXIT -ne 0 ]; then
    if [ -d "${OUTPUT_DIR}" ] && ls "${OUTPUT_DIR}"/checkpoint-*/trainer_state.json >/dev/null 2>&1; then
      echo "[${CURRENT}/${TOTAL}] WARNING: 训练退出码=${TRAIN_EXIT}，但 checkpoint 已保存，继续"
    else
      echo "[${CURRENT}/${TOTAL}] ERROR: 训练失败 (exit=${TRAIN_EXIT})，无 checkpoint"
      exit $TRAIN_EXIT
    fi
  fi
}

# ======================= 主循环 =======================
TOTAL=${#MODELS[@]}
CURRENT=0

for entry in "${MODELS[@]}"; do
  CURRENT=$((CURRENT + 1))
  IFS='|' read -r MODEL_PATH ACCEL_CONFIG OUTPUT_SUBDIR GPU_UTIL MAX_COMP_LEN VLLM_MAX_LEN <<< "$entry"
  OUTPUT_DIR="${MODEL_PATH}/${OUTPUT_SUBDIR}"
  MODEL_NAME=$(basename "$MODEL_PATH")

  echo ""
  echo "============================================================"
  echo "[${CURRENT}/${TOTAL}] Intuitor 训练: ${MODEL_NAME}"
  echo "   模型: ${MODEL_PATH}"
  echo "   输出: ${OUTPUT_DIR}"
  echo "============================================================"

  if [ -d "${OUTPUT_DIR}" ] && ls "${OUTPUT_DIR}"/checkpoint-*/trainer_state.json >/dev/null 2>&1; then
    echo "[${CURRENT}/${TOTAL}] SKIP  已有 checkpoint"
    continue
  fi

  rm -rf "${OUTPUT_DIR}"
  YAML_FILE="${TMP_DIR}/config_${MODEL_NAME}.yaml"
  _gen_yaml "${MODEL_PATH}" "${OUTPUT_DIR}" "${GPU_UTIL}" "${MAX_COMP_LEN}" "${VLLM_MAX_LEN}" > "$YAML_FILE"
  echo "   配置: ${YAML_FILE}"
  echo "   GPU util: ${GPU_UTIL}  max_comp: ${MAX_COMP_LEN}  max_model: ${VLLM_MAX_LEN}"
  echo "[${CURRENT}/${TOTAL}] 开始训练..."
  _run_training "${OUTPUT_DIR}"
  echo "[${CURRENT}/${TOTAL}] DONE → ${OUTPUT_DIR}"
done

echo ""
echo "============================================================"
echo "  Intuitor 全部训练完成！共 ${TOTAL} 个模型"
echo "============================================================"
