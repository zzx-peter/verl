set -x


gsm8k_train_path=/root/autodl-tmp/gsm8k/train.parquet
gsm8k_test_path=/root/autodl-tmp/gsm8k/test.parquet
math_train_path=/root/autodl-tmp/math/train.parquet
math_test_path=/root/autodl-tmp/math/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

clip_ratio_low=0.0003 # as recommended by the paper, see Sec. 5.1
clip_ratio_high=0.0004 # as recommended by the paper, see Sec. 5.1
clip_weight_low=0.0006
clip_weight_high=0.0
# Main and auxiliary models
# main_model=Qwen/Qwen2.5-1.5B-Instruct
# aux_model=Qwen/Qwen2.5-Math-1.5B-Instruct
main_model=/root/autodl-tmp/huggingface_cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
aux_model=/root/autodl-tmp/huggingface_cache/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35
offload=True

CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=mapo \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${main_model} \
    aux_model.enable=True \
    aux_model.model.path=${aux_model} \
    algorithm.model_source_baseline=False \
    actor_rollout_ref.actor.model_source_weighting=False \
    actor_rollout_ref.actor.model_source_clip=False \
    actor_rollout_ref.actor.model_source_performance=False \
    actor_rollout_ref.actor.use_aux_filter=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32  \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_weight_low=${clip_weight_low} \
    actor_rollout_ref.actor.clip_weight_high=${clip_weight_high} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.mapo_gamma=0.0 \
    trainer.balance_batch=False \
    trainer.critic_warmup=0 \
    trainer.logger='["tensorboard"]' \
    trainer.val_before_train=False \
    trainer.project_name='gspo_gsm8k' \
    trainer.experiment_name='qwen2.5_1.5b_Math-1.5b_baseline_new' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=3 \
    trainer.default_local_dir=/root/autodl-tmp/checkpoints/gspo_gsm8k/qwen2.5-1.5b $@
