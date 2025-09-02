set -x


gsm8k_train_path=/root/autodl-tmp/gsm8k/train.parquet
gsm8k_test_path=/root/autodl-tmp/gsm8k/test.parquet
math_train_path=/root/autodl-tmp/math/train.parquet
math_test_path=/root/autodl-tmp/math/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32  \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=True \
    trainer.critic_warmup=0 \
    trainer.logger='["tensorboard"]' \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2.5_3b_function_rm' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=/root/autodl-tmp/checkpoints/verl_grpo_example_gsm8k/qwen2.5_7b_function_rm $@
