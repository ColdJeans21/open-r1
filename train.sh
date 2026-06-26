#!/bin/bash
# 自动化 Phase 2 网格搜索训练脚本
# 用法: bash train.sh
# 会依次执行每组 (model, alpha, beta) 配置，自动生成临时 yaml 并训练

set -e

# ===================== 公共配置 =====================
ACCELERATE_CONFIG="recipes/accelerate_configs/zero3_offload.yaml"  # 8B 单卡：ZeRO-3 + CPU offload
BASE_CONFIG="recipes/Qwen2.5-1.5B-Instruct/grpo/config_limo_phase2.yaml"
LOG_LEVEL="info"
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline
export RAY_DISABLE_METRICS_EXPORTER=1
# ===================================================

# ── 生成临时 yaml 的 Python 函数 ──
_gen_yaml() {
    python3 - "$@" << 'PYEOF'
import sys, yaml, copy

base_path = sys.argv[1]
overrides = {}
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    if v.lower() == "true":  v = True
    elif v.lower() == "false":  v = False
    else:
        try:    v = int(v)
        except:
            try:    v = float(v)
            except: pass
    overrides[k] = v

with open(base_path) as f:
    cfg = yaml.safe_load(f)

cfg.update(overrides)

out = overrides.get("_output", f"/tmp/phase2_auto_{overrides.get('phase2_alpha','x')}_{overrides.get('phase2_beta','x')}.yaml")
cfg.pop("_output", None)

with open(out, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(out)
PYEOF
}

# ── 运行单次训练 ──
run_one() {
    local model_path="$1"
    local input_json="$2"
    local alpha="$3"
    local beta="$4"
    local base_dir="$5"

    local folder="alpha${alpha}_beta${beta}"
    local output_dir="${base_dir}/${folder}"
    local metrics_json="${output_dir}/phase2_metrics.jsonl"

    echo ""
    echo "############################################################"
    echo "#  model: ${model_path}"
    echo "#  alpha: ${alpha}   beta: ${beta}"
    echo "#  output: ${output_dir}"
    echo "############################################################"

    local tmp_yaml
    tmp_yaml=$(_gen_yaml "${BASE_CONFIG}" \
        "model_name_or_path=${model_path}" \
        "phase2_input_json=${input_json}" \
        "phase2_alpha=${alpha}" \
        "phase2_beta=${beta}" \
        "phase2_output_json=${metrics_json}" \
        "output_dir=${output_dir}" \
        "_output=/tmp/phase2_$$_${alpha}_${beta}.yaml")

    ACCELERATE_LOG_LEVEL="${LOG_LEVEL}" \
    accelerate launch --config_file "${ACCELERATE_CONFIG}" \
        src/open_r1/grpo_phase2.py --config "${tmp_yaml}"

    local rc=$?
    rm -f "${tmp_yaml}"
    if [ $rc -ne 0 ]; then
        echo "ERROR: Training failed for alpha=${alpha} beta=${beta}, rc=${rc}"
    fi
    return $rc
}

# ==================================================================
#  原 4B + 1.7B 实验组（已注释）
# ==================================================================
# MODEL_4B="/gpt/work/Ryan/xz/Qwen3-4B/phase1_zero_acc_with_entropy_idea6/checkpoint-817"
# INPUT_4B="/gpt/work/Ryan/xz/Qwen3-4B/phase1_zero_acc_with_entropy_idea6/zero_acc_with_entropy.json"
# BASE_4B="/gpt/work/Ryan/xz/Qwen3-4B/idea6_beta"
# for beta in 1.0; do
#     run_one "${MODEL_4B}" "${INPUT_4B}" 2.5 "${beta}" "${BASE_4B}" || true
# done
#
# MODEL_1_7B="/gpt/work/Ryan/xz/Qwen3-1.7B"
# INPUT_1_7B="/gpt/work/Ryan/xz/Qwen3-4B/phase1_zero_acc_with_entropy_idea6/zero_acc_with_entropy.json"
# BASE_1_7B="/gpt/work/Ryan/xz/Qwen3-1.7B/idea6_beta"
# for pair in "1.5 0.5" "1.5 1.0" "2.5 0.0" "2.5 0.5" "2.5 1.0"; do
#     alpha=${pair% *}
#     beta=${pair#* }
#     run_one "${MODEL_1_7B}" "${INPUT_1_7B}" "${alpha}" "${beta}" "${BASE_1_7B}" || true
# done

# ==================================================================
#  实验组：Qwen3-8B（单卡 A800 + ZeRO-2 CPU offload）
#  需要 Phase 1 产出 zero_acc_with_entropy.json，见下方 TODO
# ==================================================================
MODEL_8B="/gpt/work/Ryan/xz/Qwen3-8B"  # TODO: 替换为 Phase 1 checkpoint
INPUT_8B="/gpt/work/Ryan/xz/Qwen3-4B/phase1_zero_acc_with_entropy_idea6/zero_acc_with_entropy.json"  # TODO: Phase 1 产出
BASE_8B="/gpt/work/Ryan/xz/Qwen3-8B/idea6_beta"

for pair in "2.5 0.0" "0.5 0.0" "3.0 0.0"; do
    alpha=${pair% *}
    beta=${pair#* }
    run_one "${MODEL_8B}" "${INPUT_8B}" "${alpha}" "${beta}" "${BASE_8B}" || true
done

echo ""
echo "========================================================"
echo "  All experiments completed!"
echo "========================================================"
