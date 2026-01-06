set -x
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN_VLLM_V1
export VLLM_USE_V1=1

export SANDBOX_FUSION_ENDPOINT="http://localhost:8080"
export WANDB_API_KEY="your wandb key here"

ray stop --force

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=code_execution \
    reward_model.url=http://127.0.0.1:8005/judge/long-batch \
    reward_model.executor=oj \
    reward_model.run_all_cases=True \
    data.train_files="[\"./syn_rl_data/xcoder_data/sorted_by_passrate/rl_tasks_easy.parquet\", \"./syn_rl_data/xcoder_data/sorted_by_passrate/part_0000.parquet\",\"./syn_rl_data/xcoder_data/sorted_by_passrate/part_0001.parquet\",\"./syn_rl_data/xcoder_data/sorted_by_passrate/part_0002.parquet\",\"./syn_rl_data/xcoder_data/sorted_by_passrate/part_0003.parquet\"]" \
    data.val_files="./real_rl_data/non_sys_prompt/test_wo_prompt.parquet" \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=2500 \
    data.max_response_length=30196 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="IIGroup/X-Coder-SFT-Qwen3-8B" \
    actor_rollout_ref.model.distilled=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=4e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=35000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=110000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=110000 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    trainer.critic_warmup=0 \
    trainer.logger=["wandb"] \
    trainer.project_name="RL-Exp" \
    trainer.experiment_name="RL-Gemini-Syn-Qwen3-1ep" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.default_local_dir=./checkpoints/x-coder-qwen3-rl \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=10 \
    trainer.test_freq=1000 \
    trainer.total_epochs=3 2>&1 | tee rl-test.log