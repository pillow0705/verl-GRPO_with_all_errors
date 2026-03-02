#!/bin/bash
# ============================================================
# 冒烟测试：Negative SFT on GSM8K
# 环境：A800服务器，1张卡，3步
# 用法：bash examples/gsm8k/run_smoke_test.sh
# ============================================================

set -e

# 激活conda环境（根据服务器实际环境修改）
# conda activate verl

# 切换到项目根目录
cd "$(dirname "$0")/../.."

# 检查数据路径
DATA_DIR=/home/shared/xzliang/data/gsm8k
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "错误：找不到训练数据 $DATA_DIR/train.parquet"
    echo "请先确认数据路径，或运行数据处理脚本"
    exit 1
fi

echo "========================================="
echo "开始冒烟测试：Negative SFT on GSM8K"
echo "模型：Qwen2.5-1.5B-Instruct"
echo "显卡：1 x A800"
echo "步数：3"
echo "========================================="

CUDA_VISIBLE_DEVICES=0 python -m verl.trainer.main_ppo \
    --config-path examples/gsm8k \
    --config-name negative_sft_smoke \
    2>&1 | tee /tmp/negative_sft_smoke.log

echo "========================================="
echo "冒烟测试完成！日志保存在 /tmp/negative_sft_smoke.log"
echo "========================================="
