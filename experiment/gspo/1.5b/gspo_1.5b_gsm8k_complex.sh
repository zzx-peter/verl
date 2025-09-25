#!/usr/bin/env bash
#SBATCH --job-name=rl-gspo-3B
#SBATCH --partition=main
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=128      # cpu-cores per task
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=500:00:00
#SBATCH --output=/rl/logs/Qwen2.5-3B/gspo/math/vllm_%x_%j.out
#SBATCH --error=/rl/logs/Qwen2.5-3B/gspo/math/vllm_%x_%j.err

set -xeuo pipefail

project_name='RL-GSPO'

adv_estimator=grpo
loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"
MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
offload=false # it's a small model, offloading will just slow-down training
rollout_engine=vllm
rollout_mode=sync # can be async to speedup large scale xps
gpu_memory_utilization=0.6
reward_manager=dapo
adv_estimator=grpo
shuffle_dataset=true
first_time_dataset_prep=true # prepare dataset

test_freq=2
save_freq=-1
total_epochs=1
total_training_steps=58
val_before_train=false

use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

clip_ratio_low=0.0003 # as recommended by the paper, see Sec. 5.1
clip_ratio_high=0.0004 # as recommended by the paper, see Sec. 5.1
train_batch_size=128
ppo_mini_batch_size=32 # maintain 4 mini-batches as recommended by the paper, see Sec. 5.1
ppo_micro_batch_size_per_gpu=8 # setup depending on your GPU memory
n_resp_per_prompt=8

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
# dapo reward manager params
enable_overlong_buffer=false # true
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# Sampling params at rollouts
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=true
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
gen_tp=1
entropy_checkpointing=true # This enables entropy recomputation specifically for the entropy calculation, lowering memory usage during training.


gsm8k_train_path=/root/autodl-tmp/gsm8k/train.parquet
gsm8k_test_path=/root/autodl-tmp/gsm8k/test.parquet

# set the paths
train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.shuffle=$shuffle_dataset \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
    trainer.logger='["tensorboard"]' \
    trainer.project_name="gspo_gsm8k" \
    trainer.experiment_name="1.5b" \
    trainer.n_gpus_per_node=1\
    trainer.nnodes=1 \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=2 \
    $@
