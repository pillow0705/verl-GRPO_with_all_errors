#!/bin/bash
# ============================================================
# 冒烟测试：Negative SFT on GSM8K
# 环境：A800服务器，1张卡，3步
# 用法：bash examples/gsm8k/run_smoke_test.sh
# ============================================================

set -e

# 激活conda环境：使用 conda run -n luffy bash examples/gsm8k/run_smoke_test.sh
# 或先 conda activate luffy 再运行本脚本

# 切换到项目根目录
cd "$(dirname "$0")/../.."

# 检查数据路径
DATA_DIR=/home/shared/xzliang/data/gsm8k/processed_boxed
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "错误：找不到训练数据 $DATA_DIR/train.parquet"
    echo "请先确认数据路径，或运行数据处理脚本"
    exit 1
fi

# verl使用vllm V1引擎（AsyncLLM），需要在vllm 0.8.x上显式开启
export VLLM_USE_V1=1

echo "========================================="
echo "开始冒烟测试：Negative SFT on GSM8K"
echo "模型：Qwen2.5-1.5B-Instruct"
echo "显卡：1 x A800"
echo "步数：3"
echo "========================================="

CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path=/home/shared/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=negative_sft \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.response_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    data.tokenizer=/home/shared/Qwen2.5-1.5B-Instruct \
    "data.train_files=$DATA_DIR/train.parquet" \
    "data.val_files=$DATA_DIR/test.parquet" \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.return_raw_chat=true \
    trainer.total_training_steps=3 \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.project_name=negative_sft_research \
    trainer.experiment_name=smoke_test_gsm8k \
    'trainer.logger=[console]' \
    trainer.val_before_train=false \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.default_local_dir=/tmp/negative_sft_smoke_ckpt \
    2>&1 | tee /tmp/negative_sft_smoke.log

echo "========================================="
echo "冒烟测试完成！日志保存在 /tmp/negative_sft_smoke.log"
echo "========================================="
