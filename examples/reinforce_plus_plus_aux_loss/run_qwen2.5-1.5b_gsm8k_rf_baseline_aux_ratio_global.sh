set -x

# Paths (edit to your dataset/model locations)
gsm8k_train_path=/root/autodl-tmp/gsm8k/train.parquet
gsm8k_test_path=/root/autodl-tmp/gsm8k/test.parquet
math_train_path=/root/autodl-tmp/math/train.parquet
math_test_path=/root/autodl-tmp/math/test.parquet

# Main and auxiliary models
main_model=Qwen/Qwen2.5-1.5B-Instruct
aux_model=Qwen/Qwen2.5-3B-Instruct

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus_baseline \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=k2 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$main_model \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16  \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.model_source_weighting_method=global \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    aux_model.enable=True \
    aux_model.model.path=$aux_model \
    trainer.critic_warmup=0 \
    trainer.logger='["tensorboard"]' \
    trainer.val_before_train=True \
    trainer.project_name='qwen2.5_1.5b_3b_gsm8k_n8' \
    trainer.experiment_name='aux_model_global_seq_ratio_loss' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=3 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=/root/autodl-tmp/checkpoints/verl_reinforce++/qwen2.5_1.5b_3b_gsm8k $@