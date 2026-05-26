#!/bin/bash
# GRPO训练脚本：使用 accelerate 启动 Qwen2.5-1.5B-Instruct 的 GRPO 训练
# 执行前需赋予权限：chmod +x run_grpo_training.sh
# 运行方式：./run_grpo_training.sh

# ===================== 可修改配置（根据需求调整）=====================
ACCELERATE_CONFIG="recipes/accelerate_configs/zero2.yaml"  # accelerate 配置文件路径
GRPO_CONFIG="recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo1.yaml"  # GRPO 配置文件路径
BASE_PATH=""
LORA_OUTPUT_DIR=""
MERGED_OUTPUT_DIR=""
LOG_LEVEL="info"  # 日志级别（info/warning/error）
export WANDB_MODE=offline
export RAY_DISABLE_METRICS_EXPORTER=1
export CUDA_VISIBLE_DEVICES=1
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

# python3 -c "
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
# import os

# base_path = '/gpt/work/Ryan/xz/Qwen3-8B'
# lora_path = '$LORA_OUTPUT_DIR'
# output_path = '$MERGED_OUTPUT_DIR'

# if not os.path.exists(lora_path):
#     print(f'❌ 找不到 LoRA 路径: {lora_path}，请检查训练是否成功生成了权重。')
#     exit(1)

# print('加载底座模型...')
# base_model = AutoModelForCausalLM.from_pretrained(
#     base_path, torch_dtype=torch.bfloat16, device_map='cpu'
# )
# print('加载 Tokenizer...')
# tokenizer = AutoTokenizer.from_pretrained(base_path)

# print('合并参数中...')
# model = PeftModel.from_pretrained(base_model, lora_path)
# model = model.merge_and_unload()

# print('保存合并后的模型...')
# model.save_pretrained(output_path)
# tokenizer.save_pretrained(output_path)
# "

# echo "✅ [3/3] 合并完成！全新模型已保存至：$MERGED_OUTPUT_DIR"
# echo "🎉 全流程结束，你可以直接去跑评测了！"

# 训练完成提示
if [ $? -eq 0 ]; then
    echo "==================== 训练成功 ===================="
else
    echo "==================== 训练失败 ===================="
    exit 1
fi