#!/bin/bash

# Qwen3-32B LoRA training script using Megatron backend with GRPO.
# Requires 8x GPUs (e.g. 8x H100/A100 80GB) with TP=4 for training + colocated rollout.

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${REPO_ROOT}/scripts/models/qwen3-32B.sh"

HF_CHECKPOINT=${HF_CHECKPOINT:-"/root/models/Qwen3-32B"}
DATASET_PATH=${DATASET_PATH:-"/root/miles/training_prompts.jsonl"}
SAVE_PATH=${SAVE_PATH:-"/root/checkpoints/qwen3-32B-lora"}

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --megatron-to-hf-mode bridge
   --save "${SAVE_PATH}"
   --save-interval 5
)

LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 32
   --lora-dropout 0.0
   --target-modules "all-linear"
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data "${DATASET_PATH}"
   --input-key prompt
   --label-key label
   --tool-key tools
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 15
   --rollout-batch-size 128
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1
   --global-batch-size 1024
   --balance-data
)

MULTI_TURN_ARGS=(
   --custom-generate-function-path calculator_tool.generate
   --custom-rm-path calculator_tool.reward_func
)

EVAL_ARGS=(
)

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-host https://wandb.ai/
   # --wandb-team miles-lora
   # --wandb-project miles-lora-megatron
   # --wandb-group qwen3-32B-lora
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-enable-deterministic-inference
   --sglang-attention-backend flashinfer
   --deterministic-mode
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus $GPUS_PER_NODE --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${SCRIPT_DIR}:/root/miles\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_ALGO\": \"Ring\",
    \"NVTE_ALLOW_NONDETERMINISTIC_ALGO\": \"0\",
    \"CUBLAS_WORKSPACE_CONFIG\": \":4096:8\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node $GPUS_PER_NODE \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${LORA_ARGS[@]} \
   ${MULTI_TURN_ARGS[@]}
