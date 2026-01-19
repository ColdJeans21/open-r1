#!/bin/bash
# 模型评估脚本：使用 lighteval 评估 Qwen3-0.6B-Math-Expert 在 gsm8k 数据集上的性能
# 执行前需赋予权限：chmod +x lighteval_gsm8k_eval.sh
# 运行方式：./lighteval_gsm8k_eval.sh

# ===================== 可修改配置（根据需求调整）=====================
NUM_GPUS=1                          # 使用的GPU数量
MODEL_PATH="/gpt/work/Ryan/xz/Qwen3-0.6B-idea4/to_25.8epoch_new/checkpoint-500"  # 模型路径
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
#!/bin/bash
# 模型评估脚本：使用 lighteval 评估 Qwen3-0.6B-Math-Expert 在 gsm8k 数据集上的性能
# 执行前需赋予权限：chmod +x lighteval_gsm8k_eval.sh
# 运行方式：./lighteval_gsm8k_eval.sh

# ===================== 可修改配置（根据需求调整）=====================
# NUM_GPUS=1                      # 使用的GPU数量
# MODEL_PATH="/gpt/work/Ryan/xz/Qwen3-0.6B-Math-Expert"  # 模型路径
# MAX_MODEL_LENGTH=8192           # 模型最大序列长度
# MAX_NEW_TOKENS=4096             # 【修改建议】6192可能太极限，建议留点空间给Prompt，4096对GSM8K足够了
# GPU_MEM_UTIL=0.6                # 【关键修改】调高到 0.6 或 0.7，防止长推理时 KV Cache 显存不足导致截断
# TASK=gsm8k                      # 评估任务
# OUTPUT_DIR="${MODEL_PATH}/eval" # 评估结果输出目录

# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# # ====================================================================

# # 【关键修改】构建模型参数
# # 1. 添加 trust_remote_code=True 以支持 Qwen 架构代码
# # 2. 修改 temperature:0 (贪婪解码，数学任务标准)
# # 3. 移除 top_p (贪婪解码不需要)
# MODEL_ARGS="model_name=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True,data_parallel_size=${NUM_GPUS},max_model_length=${MAX_MODEL_LENGTH},gpu_memory_utilization=${GPU_MEM_UTIL},generation_parameters={max_new_tokens:${MAX_NEW_TOKENS},temperature:0,repetition_penalty:1.0}"

# # 创建输出目录
# if [ ! -d "${OUTPUT_DIR}" ]; then
#     echo "创建输出目录：${OUTPUT_DIR}"
#     mkdir -p "${OUTPUT_DIR}"
# fi

# # 执行评估命令
# echo "开始执行评估任务：${TASK}"
# echo "模型路径：${MODEL_PATH}"
# echo "GPU数量：${NUM_GPUS}"
# echo "输出目录：${OUTPUT_DIR}"
# echo "==================== 执行命令 ===================="

# # 注意：gsm8k 任务通常不需要额外的 system prompt，但为了保险起见，
# # 如果模型依然输出奇怪格式，可以尝试在 arguments 中寻找 system-prompt 相关设置（视 lighteval 版本而定）
# # 这里保持基础命令，依靠 temperature=0 和足够的显存来修复。

# lighteval vllm "${MODEL_ARGS}" "lighteval|${TASK}|0|0" \
#     --use-chat-template \
#     --output-dir "${OUTPUT_DIR}" 

# # 评估完成提示
# if [ $? -eq 0 ]; then
#     echo "==================== 评估成功 ===================="
#     echo "评估结果已保存到：${OUTPUT_DIR}"
# else
#     echo "==================== 评估失败 ===================="
#     # 打印部分日志以供调试
#     echo "请检查上方日志中的 Python Traceback 错误信息。"
#     exit 1
# fi


# git checkout -b my-update
# git add .
# git commit -m "插入的prompt不参与优势计算（完整版）"
# git checkout main
# git merge my-update
# git push origin main