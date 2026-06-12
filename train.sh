#!/bin/bash
# 两阶段训练入口
# Phase 1: GRPO 训练 + 零acc 熵采集
# Phase 2: 熵截断 hint 重采样 + GRPO 训练
# 运行方式：bash train.sh

# ===================== 可修改配置 =====================
ACCELERATE_CONFIG="recipes/accelerate_configs/zero2.yaml"
PHASE1_CONFIG="recipes/Qwen2.5-1.5B-Instruct/grpo/config_limo_phase1.yaml"
PHASE2_CONFIG="recipes/Qwen2.5-1.5B-Instruct/grpo/config_limo_phase2.yaml"
LOG_LEVEL="info"
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline
export RAY_DISABLE_METRICS_EXPORTER=1
# ====================================================

# echo "========================================================"
# echo "  Phase 1: GRPO Training + Zero-Acc Entropy Collection"
# echo "========================================================"

# ACCELERATE_LOG_LEVEL="${LOG_LEVEL}" \
# accelerate launch --config_file "${ACCELERATE_CONFIG}" \
#     src/open_r1/grpo.py --config "${PHASE1_CONFIG}" \
#     --vllm_mode colocate

# if [ $? -ne 0 ]; then
#     echo "Phase 1 failed, stopping."
#     exit 1
# fi

echo ""
echo "========================================================"
echo "  Phase 2: Hint Resample + GRPO Training"
echo "========================================================"

ACCELERATE_LOG_LEVEL="${LOG_LEVEL}" \
accelerate launch --config_file "${ACCELERATE_CONFIG}" \
    src/open_r1/grpo_phase2.py --config "${PHASE2_CONFIG}" \
    --vllm_mode colocate

if [ $? -eq 0 ]; then
    echo "========================================================"
    echo "  All phases complete!"
    echo "========================================================"
else
    echo "========================================================"
    echo "  Phase 2 failed!"
    echo "========================================================"
    exit 1
fi
