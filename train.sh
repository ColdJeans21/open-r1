#!/bin/bash
# 原始 GRPO 对照实验 — 自动化训练 1.7B / 4B / 8B
# 数据集: GAIR/LIMO + zero_acc_questions 合并
# 单卡 GPU 1

set -e

export CUDA_VISIBLE_DEVICES=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 使用 open-r1 venv
VENV_BIN="${SCRIPT_DIR}/openr1/bin"
export PATH="${VENV_BIN}:${PATH}"
TMP_DIR=$(mktemp -d -t grpo_origin_XXXX)
trap "rm -rf $TMP_DIR" EXIT

# ======================= 模型配置 =======================
# 格式: "模型路径|accelerate_config|输出子目录"
# 格式: "模型路径|accelerate_config|输出子目录|gpu_memory_util|max_completion_len|vllm_max_model_len"
MODELS=(
  "/gpt/work/Ryan/xz/Qwen3-1.7B|zero2.yaml|origin_GRPO|0.35|6120|8192"
  "/gpt/work/Ryan/xz/Qwen3-4B|zero3_offload.yaml|origin_GRPO|0.55|2048|3072"
  "/gpt/work/Ryan/xz/Qwen3-8B|zero3_offload.yaml|origin_GRPO|0.75|2048|3072"
)

# ======================= 生成临时 YAML =======================
_gen_yaml() {
  local model_path="$1"
  local output_dir="$2"
  local vllm_mode="${3:-server}"
  local gpu_util="$4"
  local max_comp_len="$5"
  local vllm_max_len="$6"

  python3 - "$model_path" "$output_dir" "$vllm_mode" "$gpu_util" "$max_comp_len" "$vllm_max_len" <<'PYEOF'
import sys, yaml

model_path = sys.argv[1]
output_dir = sys.argv[2]
vllm_mode = sys.argv[3]
gpu_util = float(sys.argv[4])
max_comp_len = int(sys.argv[5])
vllm_max_len = int(sys.argv[6])

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
    "system_prompt": (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, "
        "and put your final answer within \\boxed{} . The reasoning process and answer are enclosed within "
        "<think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think> <answer> answer here </answer>."
    ),
    "bf16": True,
    "use_vllm": True,
    "vllm_gpu_memory_utilization": gpu_util,
    "vllm_max_model_len": vllm_max_len,
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
    "max_completion_length": max_comp_len,
    "num_train_epochs": 1,
    "num_generations": 4,
    "per_device_eval_batch_size": 1,
    "per_device_train_batch_size": 1,
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "seed": 50,
    "warmup_ratio": 0.1,
    "reward_funcs": ["accuracy", "format", "tag_count"],
    "reward_weights": [1.0, 1.0, 1.0],
    "output_dir": output_dir,
    "overwrite_output_dir": False,
    "push_to_hub": False,
    "wandb_log_unique_prompts": False,
    "use_peft": False,
    "report_to": ["wandb"],
}
if vllm_mode == "server":
    config["vllm_server_port"] = 18888

yaml_str = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
print(yaml_str)
PYEOF
}

# ======================= 训练执行 =======================
_run_training() {
  local MODEL_NAME="$1"
  local MODEL_PATH="$2"
  local ACCEL_CONFIG="$3"
  local OUTPUT_DIR="$4"
  local EXTRA_ARGS="$5"

  cd "${SCRIPT_DIR}"
  set +e
  accelerate launch \
    --config_file "recipes/accelerate_configs/${ACCEL_CONFIG}" \
    --num_processes=1 \
    src/open_r1/grpo.py \
    --config "${YAML_FILE}" \
    ${EXTRA_ARGS}
  local TRAIN_EXIT=$?
  set -e

  if [ $TRAIN_EXIT -ne 0 ]; then
    if [ -d "${OUTPUT_DIR}" ] && ls "${OUTPUT_DIR}"/checkpoint-*/trainer_state.json >/dev/null 2>&1; then
      echo "[${CURRENT}/${TOTAL}] WARNING: 训练退出码=${TRAIN_EXIT} (可能 save OOM)，但 checkpoint 已保存，继续"
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
  echo "[${CURRENT}/${TOTAL}] 训练: ${MODEL_NAME}"
  echo "   模型路径: ${MODEL_PATH}"
  echo "   Accelerate config: ${ACCEL_CONFIG}"
  echo "   输出目录: ${OUTPUT_DIR}"
  echo "============================================================"

  # 跳过已完成训练的
  if [ -d "${OUTPUT_DIR}" ] && ls "${OUTPUT_DIR}"/checkpoint-*/trainer_state.json >/dev/null 2>&1; then
    echo "[${CURRENT}/${TOTAL}] SKIP  ${MODEL_NAME}  已有 checkpoint"
    continue
  fi

  # 清理失败残留
  rm -rf "${OUTPUT_DIR}"

  # ── colocate 模式（server 模式与 vLLM 0.8.5 不兼容）──
  YAML_FILE="${TMP_DIR}/config_${MODEL_NAME}.yaml"
  _gen_yaml "${MODEL_PATH}" "${OUTPUT_DIR}" "colocate" "${GPU_UTIL}" "${MAX_COMP_LEN}" "${VLLM_MAX_LEN}" > "$YAML_FILE"
  echo "   配置文件: ${YAML_FILE}"
  echo "   GPU util: ${GPU_UTIL}  max_completion: ${MAX_COMP_LEN}  max_model_len: ${VLLM_MAX_LEN}"
  echo "[${CURRENT}/${TOTAL}] 开始训练 ${MODEL_NAME}..."
  _run_training "${MODEL_NAME}" "${MODEL_PATH}" "${ACCEL_CONFIG}" "${OUTPUT_DIR}" "--vllm_mode colocate"

  echo "[${CURRENT}/${TOTAL}] DONE → ${OUTPUT_DIR}"
done

echo ""
echo "============================================================"
echo "  全部训练完成！共 ${TOTAL} 个模型"
echo "============================================================"
