#!/bin/bash
# 模型评估脚本：使用 lighteval 评估 Qwen3-0.6B-Math-Expert 在 gsm8k 数据集上的性能
# 执行前需赋予权限：chmod +x lighteval_gsm8k_eval.sh
# 运行方式：./lighteval_gsm8k_eval.sh

# ===================== 可修改配置（根据需求调整）=====================
NUM_GPUS=1                          # 使用的GPU数量
MODEL_PATH="/home/gml/xz/Qwen3-0.6B-Math-Expert-idea3/to_2000steps_8.6epoch/checkpoint-500"  # 模型路径
MAX_MODEL_LENGTH=8192               # 模型最大序列长度
MAX_NEW_TOKENS=6192                 # 生成新token的最大数量（需满足：输入长度+6192 ≤8192）
GPU_MEM_UTIL=0.3                    # GPU显存利用率（避免显存溢出）
TASK=gsm8k # mathqa # gsm8k                          # 评估任务（数据集）
OUTPUT_DIR="${MODEL_PATH}/eval"     # 评估结果输出目录
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# ====================================================================

# 构建模型参数（无需修改，自动读取上方配置）
MODEL_ARGS="model_name=${MODEL_PATH},dtype=bfloat16,data_parallel_size=${NUM_GPUS},max_model_length=${MAX_MODEL_LENGTH},gpu_memory_utilization=${GPU_MEM_UTIL},generation_parameters={max_new_tokens:${MAX_NEW_TOKENS},temperature:0.6,top_p:0.95}"

# 创建输出目录（若不存在）
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "创建输出目录：${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
fi

# 执行评估命令
echo "开始执行评估任务：${TASK}"
echo "模型路径：${MODEL_PATH}"
echo "GPU数量：${NUM_GPUS}"
echo "输出目录：${OUTPUT_DIR}"
echo "==================== 执行命令 ===================="
lighteval vllm "${MODEL_ARGS}" "lighteval|${TASK}|0|0" \
    --use-chat-template \
    --output-dir "${OUTPUT_DIR}"

# 评估完成提示
if [ $? -eq 0 ]; then
    echo "==================== 评估成功 ===================="
    echo "评估结果已保存到：${OUTPUT_DIR}"
else
    echo "==================== 评估失败 ===================="
    exit 1
fi