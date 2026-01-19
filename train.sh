#!/bin/bash
# GRPO训练脚本：使用 accelerate 启动 Qwen2.5-1.5B-Instruct 的 GRPO 训练
# 执行前需赋予权限：chmod +x run_grpo_training.sh
# 运行方式：./run_grpo_training.sh

# ===================== 可修改配置（根据需求调整）=====================
ACCELERATE_CONFIG="recipes/accelerate_configs/zero3.yaml"  # accelerate 配置文件路径
GRPO_CONFIG="recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo1.yaml"  # GRPO 配置文件路径
LOG_LEVEL="info"  # 日志级别（info/warning/error）
export WANDB_MODE=offline
export RAY_DISABLE_METRICS_EXPORTER=1
# ====================================================================

# 执行训练命令
echo "开始 GRPO 训练任务"
echo "accelerate 配置：${ACCELERATE_CONFIG}"
echo "GRPO 配置：${GRPO_CONFIG}"
echo "日志级别：${LOG_LEVEL}"
echo "==================== 执行命令 ===================="
ACCELERATE_LOG_LEVEL="${LOG_LEVEL}" \
accelerate launch --config_file "${ACCELERATE_CONFIG}" \
    src/open_r1/grpo.py --config "${GRPO_CONFIG}" \
    --vllm_mode colocate

# 训练完成提示
if [ $? -eq 0 ]; then
    echo "==================== 训练成功 ===================="
else
    echo "==================== 训练失败 ===================="
    exit 1
fi