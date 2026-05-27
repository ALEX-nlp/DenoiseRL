#!/usr/bin/env bash
set -euxo pipefail
export WANDB_MODE=offline
version="v1.0"

# -----------------------------------------------------------------------------
# denoise specific knobs
# -----------------------------------------------------------------------------
# Each prompt produces N "main" rollouts and K "sub" rollouts. Sub rollouts use
# the first ``part_response_ratio`` (by tokens) of a row drawn from
# ``wrong_answer_with_boxed`` as a partial assistant prefix; the model continues
# from that prefix with ``continue_final_message=True``. After rollout the
# partial-wrong prefix is folded into the response window per ``partial_mode``:
#   * "shift"   - response width = R; trailing p_i tokens truncated so
#                 partial_wrong + kept_continuation <= R (length-fair). partial_wrong
#                 gets response_mask = 1 (gradient flows through the off-policy
#                 prefix). Max signal but potentially less stable.
#   * "cutdown" - response width = R; same truncation as "shift". partial_wrong gets
#                 response_mask = 0 (no gradient on the prefix). More stable, less
#                 signal.
#   * "none"    - response width = R + max_partial_len; NO truncation (per-row
#                 output is p_i + R; NOT length-fair vs main rollouts). partial_wrong
#                 gets response_mask = 0; all R generated tokens preserved.
n_resp_per_prompt=12
sub_rollout_k=4
partial_mode=cutdown

# -----------------------------------------------------------------------------
# part_response_ratio sampling strategy
# -----------------------------------------------------------------------------
# Each K sub-rollout draws its own ``part_response_ratio`` (the fraction of the
# wrong solution tokens kept as a partial assistant prefix). The Python side
# resamples on every sub-rollout slot according to ``part_response_ratio_strategy``:
#   * "fixed"   - constant value ``part_response_ratio_fixed``.
#   * "normal"  - sample ratio ~ N(mean, std), then clip to
#                 [part_response_ratio_low, part_response_ratio_high].
#   * "uniform" - sample ratio ~ U(part_response_ratio_low, part_response_ratio_high).
# Bounds must satisfy 0 < low <= high <= 1.
part_response_ratio_strategy="fixed"   # "fixed" | "normal" | "uniform"
part_response_ratio_fixed=0.2
part_response_ratio_mean=0.5
part_response_ratio_std=0.2
part_response_ratio_low=0.2
part_response_ratio_high=0.8

# Compact tag used in run/exp names so different strategies produce distinct ids.
case "${part_response_ratio_strategy}" in
    fixed)
        ratio_tag="fix${part_response_ratio_fixed}"
        ;;
    normal)
        ratio_tag="norm-m${part_response_ratio_mean}-s${part_response_ratio_std}-lo${part_response_ratio_low}-hi${part_response_ratio_high}"
        ;;
    uniform)
        ratio_tag="uni-lo${part_response_ratio_low}-hi${part_response_ratio_high}"
        ;;
    *)
        echo "Unknown part_response_ratio_strategy: ${part_response_ratio_strategy}" >&2
        exit 1
        ;;
esac

# Split main vs. sub by actual partial prefix (partial_response_len > 0).
# Fallback sub slots (no wrong solution) are treated as main.
sub_rollout_separate_adv_uid=False
sub_rollout_separate_loss_group=False
sub_rollout_loss_multiplier=1.0

# GRPO grouping: per-problem (all N+K trajectories of one problem share a uid).
use_problem_id_as_uid=True
use_same_uid=False


# -----------------------------------------------------------------------------
# Model / cluster
# -----------------------------------------------------------------------------
model_name="Qwen3-4B-Base"
offload=True
ref_offload=True
num_gpus=8
tensor_model_parallel_size=2

# -----------------------------------------------------------------------------
# Training schedule
# -----------------------------------------------------------------------------
epoch=1000
project_name='V1.0 Denoise 4B'

lr_warmup_steps=0
lr=1e-6
test_and_save_freq=40
train_prompt_bsz=16
# Total responses per batch = train_prompt_bsz * (n_resp_per_prompt + sub_rollout_k).
# We keep the actor on-policy by sizing the PPO mini-batch to match the full batch.
train_prompt_mini_bsz=16
force_on_policy=True

wandb_run_id="${version}_${sub_rollout_k}_${ratio_tag}_adv-split-${sub_rollout_separate_adv_uid}_loss-split-${sub_rollout_separate_loss_group}_sub-mult-${sub_rollout_loss_multiplier}_denoise_v1.0_partial_wrong"
exp_name=${exp_name:-"debug_${version}-partial-wrong-k-${sub_rollout_k}-ratio-${ratio_tag}-adv-split-${sub_rollout_separate_adv_uid}-loss-split-${sub_rollout_separate_loss_group}-sub-mult-${sub_rollout_loss_multiplier}-problem-id-${use_problem_id_as_uid}-response-same-${use_same_uid}-model-${model_name}-lr-${lr}-bsz-${train_prompt_bsz}-n_resp-${n_resp_per_prompt}-mini-${train_prompt_mini_bsz}"}

adv_estimator=grpo

gpu_memory_utilization=0.7
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
norm_adv_by_std_in_grpo=True

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 8))
max_response_length=$((1024 * 4))

loss_agg_mode="token-mean"
enable_filter_groups=False
filter_groups_metric=acc

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
RAY_DATA_HOME=${RAY_DATA_HOME:-"."}
MODEL_PATH=${MODEL_PATH:-"../Qwen/${model_name}"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"./data/MATH7500.with_wrong_boxed.qwen2.5-1.5b.with_wrong_boxed.parquet"}
TEST_FILE=${TEST_FILE:-["./data/aime25_test.parquet","./data/bbeh_data.parquet","./data/MATH500-test.parquet","./data/amc23_test.parquet","./data/aime24_test.parquet","./data/MMLU-Pro-Valid.parquet"]}

# -----------------------------------------------------------------------------
# Sampling / engine
# -----------------------------------------------------------------------------
temperature=1.0
top_p=1.0
top_k=-1

val_temperature=0.6
val_top_p=0.95

sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((( max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$((( max_prompt_length + max_response_length)))
max_num_gen_batches=100

PYTHONUNBUFFERED=1 python3 -m recipe.denoise.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=512 \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
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
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
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
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    reward_model.reward_manager=naive \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${num_gpus}" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=${test_and_save_freq} \
    trainer.save_freq=${test_and_save_freq} \
    trainer.total_epochs=${epoch} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +trainer.max_actor_ckpt_to_keep=1 \
    +trainer.sub_rollout_k=${sub_rollout_k} \
    +trainer.part_response_ratio_strategy=${part_response_ratio_strategy} \
    +trainer.part_response_ratio_fixed=${part_response_ratio_fixed} \
    +trainer.part_response_ratio_mean=${part_response_ratio_mean} \
    +trainer.part_response_ratio_std=${part_response_ratio_std} \
    +trainer.part_response_ratio_low=${part_response_ratio_low} \
    +trainer.part_response_ratio_high=${part_response_ratio_high} \
    +trainer.partial_mode=${partial_mode} \
    +trainer.use_problem_id_as_uid=${use_problem_id_as_uid} \
    +trainer.use_same_uid=${use_same_uid} \
    +trainer.sub_rollout_separate_adv_uid=${sub_rollout_separate_adv_uid} \
    +trainer.sub_rollout_separate_loss_group=${sub_rollout_separate_loss_group} \
    +trainer.sub_rollout_loss_multiplier=${sub_rollout_loss_multiplier} \
    +trainer.wandb_run_id=${wandb_run_id}
