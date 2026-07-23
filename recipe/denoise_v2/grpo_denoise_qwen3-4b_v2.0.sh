#!/usr/bin/env bash
set -euxo pipefail

export WANDB_MODE=${WANDB_MODE:-offline}

# DenoiseRL v2 invariants: one active prompt batch, zero clean/main slots, and
# exactly 16 fixed-rho noise rollouts per prompt.
n_resp_per_prompt=0
sub_rollout_k=16
effective_rollout_n=16

# Every prompt owns an independent rho. The first active batch starts at zero;
# replacements inherit the post-update mean rho N of the previous active batch.
v2_initial_rho=${v2_initial_rho:-0.0}
v2_min_rho=${v2_min_rho:-0.0}
v2_max_rho=${v2_max_rho:-0.5}
v2_target_accuracy=${v2_target_accuracy:-0.8}
v2_alpha=${v2_alpha:-0.1}
v2_history_window=${v2_history_window:-5}
v2_min_history=${v2_min_history:-2}
# A sample is stable only inside the two-sided band:
# abs(recent_rho_slope) <= v2_slope_threshold.
v2_slope_threshold=${v2_slope_threshold:-0.01}

# Correct rollouts keep full reward before the final length cache, then linearly
# decrease to correct_length_reward_min_factor at max_response_length.
correct_length_reward_enabled=${correct_length_reward_enabled:-True}
correct_length_reward_cache_length=${correct_length_reward_cache_length:-1024}
correct_length_reward_min_factor=${correct_length_reward_min_factor:-0.0}
case "${correct_length_reward_enabled}" in
    True|true|1)
        length_reward_tag="len${correct_length_reward_min_factor}-cache${correct_length_reward_cache_length}"
        ;;
    False|false|0)
        length_reward_tag="nolen"
        ;;
    *)
        echo "correct_length_reward_enabled must be True or False, got: ${correct_length_reward_enabled}" >&2
        exit 1
        ;;
esac

# Model / cluster.
model_name=${model_name:-Qwen3-4B-Base}
offload=${offload:-True}
ref_offload=${ref_offload:-True}
num_gpus=${num_gpus:-4}
tensor_model_parallel_size=${tensor_model_parallel_size:-1}
sp_size=${sp_size:-1}

# GRPO schedule.
epoch=${epoch:-10000}
project_name=${project_name:-DenoiseRL-v2-4B}
lr=${lr:-1e-6}
lr_warmup_steps=${lr_warmup_steps:-0}
test_and_save_freq=${test_and_save_freq:-40}
train_prompt_bsz=${train_prompt_bsz:-16}
train_prompt_mini_bsz=${train_prompt_mini_bsz:-16}
force_on_policy=${force_on_policy:-True}

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
norm_adv_by_std_in_grpo=True
clip_ratio_low=0.2
clip_ratio_high=0.2
loss_agg_mode=token-mean

max_prompt_length=${max_prompt_length:-8192}
max_response_length=${max_response_length:-4096}
gpu_memory_utilization=${gpu_memory_utilization:-0.7}
use_dynamic_bsz=${use_dynamic_bsz:-True}
actor_ppo_max_token_len=$((2 * (max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$((2 * (max_prompt_length + max_response_length)))

# Data stays in file order. Python first removes rows without a usable
# wrong_answer_with_boxed, treats the remaining rows as one pool, and explicitly
# selects active indices beginning with [0, train_prompt_bsz).
RAY_DATA_HOME=${RAY_DATA_HOME:-.}
MODEL_PATH=${MODEL_PATH:-../Model/Qwen/${model_name}}
TRAIN_FILE=${TRAIN_FILE:-./data/MATH7500.with_wrong_boxed.qwen2.5-1.5b.parquet}
TEST_FILE=${TEST_FILE:-'["./data/aime25_test.parquet","./data/bbeh_data.parquet","./data/MATH500-test.parquet","./data/amc23_test.parquet","./data/aime24_test.parquet","./data/MMLU-Pro-Valid.parquet"]'}

run_tag="rho${v2_initial_rho}-${v2_max_rho}_target${v2_target_accuracy}_alpha${v2_alpha}_window${v2_history_window}_slope${v2_slope_threshold}_${length_reward_tag}"
experiment_name=${experiment_name:-"grpo-denoise-v2-${model_name}-bsz${train_prompt_bsz}-k16-${run_tag}"}
wandb_run_id=${wandb_run_id:-${experiment_name}}
CKPTS_DIR=${CKPTS_DIR:-${RAY_DATA_HOME}/ckpts/${project_name}/${experiment_name}}

temperature=${temperature:-1.0}
top_p=${top_p:-1.0}
top_k=${top_k:--1}
val_temperature=${val_temperature:-0.6}
val_top_p=${val_top_p:-0.95}

PYTHONUNBUFFERED=1 python3 -m recipe.denoise_v2.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.shuffle=False \
    data.dataloader_num_workers=0 \
    data.truncation=left \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=512 \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.rollout_n=${effective_rollout_n} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0.0 \
    +actor_rollout_ref.model.override_config.embd_pdrop=0.0 \
    +actor_rollout_ref.model.override_config.resid_pdrop=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    +actor_rollout_ref.actor.force_on_policy=${force_on_policy} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    algorithm.filter_groups.enable=False \
    reward_model.reward_manager=naive \
    trainer.logger="['console','wandb']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=${num_gpus} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=${test_and_save_freq} \
    trainer.save_freq=${test_and_save_freq} \
    trainer.total_epochs=${epoch} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +trainer.max_actor_ckpt_to_keep=1 \
    +trainer.use_dapo=False \
    +trainer.sub_rollout_k=${sub_rollout_k} \
    +trainer.noise_source=partial_wrong \
    +trainer.partial_wrong_cut_strategy=token \
    +trainer.part_response_ratio_strategy=fixed \
    +trainer.partial_mode=cutdown \
    +trainer.use_problem_id_as_uid=True \
    +trainer.use_same_uid=False \
    +trainer.sub_rollout_separate_adv_uid=False \
    +trainer.sub_rollout_separate_loss_group=False \
    +trainer.sub_rollout_loss_multiplier=1.0 \
    +trainer.v2_curriculum_enabled=True \
    +trainer.v2_initial_rho=${v2_initial_rho} \
    +trainer.v2_min_rho=${v2_min_rho} \
    +trainer.v2_max_rho=${v2_max_rho} \
    +trainer.v2_target_accuracy=${v2_target_accuracy} \
    +trainer.v2_alpha=${v2_alpha} \
    +trainer.v2_history_window=${v2_history_window} \
    +trainer.v2_min_history=${v2_min_history} \
    +trainer.v2_slope_threshold=${v2_slope_threshold} \
    +trainer.correct_length_reward_enabled=${correct_length_reward_enabled} \
    +trainer.correct_length_reward_cache_length=${correct_length_reward_cache_length} \
    +trainer.correct_length_reward_min_factor=${correct_length_reward_min_factor} \
    +trainer.wandb_run_id="${wandb_run_id}"
