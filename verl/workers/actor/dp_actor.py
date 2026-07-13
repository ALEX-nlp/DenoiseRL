# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.actor.loss_aggregation import aggregation_mass, micro_batch_aggregation_scale
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    @staticmethod
    def _loss_group_keys(loss_group_id, sample_loss_multiplier: torch.Tensor) -> list[object]:
        """Build stable per-row group keys for both explicit and legacy grouping."""
        if loss_group_id is not None:
            return [str(value) if value is not None else "__none__" for value in loss_group_id]
        return [float(value) for value in sample_loss_multiplier.detach().cpu().tolist()]

    @staticmethod
    def _mask_aggregation_counts(loss_mask: torch.Tensor) -> tuple[float, float]:
        token_count = float(loss_mask.detach().sum().item())
        active_sequence_count = float((loss_mask.detach().sum(dim=-1) > 0).sum().item())
        return token_count, active_sequence_count

    @classmethod
    def _mask_aggregation_mass(cls, loss_mask: torch.Tensor, loss_agg_mode: str) -> float:
        token_count, active_sequence_count = cls._mask_aggregation_counts(loss_mask)
        return aggregation_mass(
            token_count=token_count,
            active_sequence_count=active_sequence_count,
            loss_agg_mode=loss_agg_mode,
        )

    @classmethod
    def _micro_batch_aggregation_scale(
        cls,
        loss_mask: torch.Tensor,
        global_mass: float,
        loss_agg_mode: str,
    ) -> float:
        token_count, active_sequence_count = cls._mask_aggregation_counts(loss_mask)
        return micro_batch_aggregation_scale(
            local_token_count=token_count,
            local_active_sequence_count=active_sequence_count,
            global_mass=global_mass,
            loss_agg_mode=loss_agg_mode,
        )

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error


        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if "loss_multiplier" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("loss_multiplier")
        if "loss_group_id" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("loss_group_id")
        if "loss_group_normalizer" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("loss_group_normalizer")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        batch_size = len(data)
        ppo_mini_batch_size = self.config.ppo_mini_batch_size
        if self.config.get('force_on_policy', False):
            # One chunk along dim=0 so len(mini_batches)==1; on_policy still requires ppo_epochs==1
            ppo_mini_batch_size = batch_size
        mini_batches = data.split(ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        print("len(mini_batches) = ", len(mini_batches))
        print("ppo_mini_batch_size = ", ppo_mini_batch_size)
        print("data.batch['responses'].shape = ", data.batch['responses'].shape)
        print("on_policy = ", on_policy)
        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                loss_agg_mode = self.config.loss_agg_mode
                loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                # GSPO hard-codes sequence-mean-token-mean inside its policy
                # loss implementation. Other policy losses use loss_agg_mode.
                policy_loss_agg_mode = (
                    "seq-mean-token-mean" if loss_mode == "gspo" else loss_agg_mode
                )

                # Record each loss group's aggregation denominator over the complete
                # mini-batch before dynamic batching splits it. A token mean must be
                # weighted by the micro-batch's share of trainable tokens, not rows.
                mini_response_mask = mini_batch.batch["response_mask"]
                mini_loss_multiplier = mini_batch.non_tensor_batch.get("loss_multiplier", None)
                if mini_loss_multiplier is None:
                    mini_multiplier_tensor = torch.ones(
                        len(mini_batch), dtype=torch.float32, device=mini_response_mask.device
                    )
                elif torch.is_tensor(mini_loss_multiplier):
                    mini_multiplier_tensor = mini_loss_multiplier.to(
                        device=mini_response_mask.device, dtype=torch.float32
                    )
                else:
                    mini_multiplier_tensor = torch.as_tensor(
                        mini_loss_multiplier, device=mini_response_mask.device, dtype=torch.float32
                    )
                mini_group_keys = self._loss_group_keys(
                    mini_batch.non_tensor_batch.get("loss_group_id", None),
                    mini_multiplier_tensor.view(-1),
                )
                mini_groups: dict[object, list[int]] = {}
                for row_idx, key in enumerate(mini_group_keys):
                    mini_groups.setdefault(key, []).append(row_idx)

                global_group_masses = {}
                global_policy_group_masses = {}
                mini_active_group_count = 0
                for key, idx_list in mini_groups.items():
                    idx = torch.as_tensor(idx_list, device=mini_response_mask.device, dtype=torch.long)
                    group_mask = mini_response_mask.index_select(0, idx)
                    if bool((group_mask.sum() > 0).item()):
                        mini_active_group_count += 1
                    global_group_masses[key] = self._mask_aggregation_mass(group_mask, loss_agg_mode)
                    global_policy_group_masses[key] = self._mask_aggregation_mass(
                        group_mask, policy_loss_agg_mode
                    )

                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    loss_multiplier = model_inputs.get("loss_multiplier", None)
                    if loss_multiplier is None:
                        sample_loss_multiplier = torch.ones(
                            advantages.shape[0], device=advantages.device, dtype=advantages.dtype
                        )
                    else:
                        if not torch.is_tensor(loss_multiplier):
                            sample_loss_multiplier = torch.as_tensor(
                                loss_multiplier, device=advantages.device, dtype=advantages.dtype
                            )
                        else:
                            sample_loss_multiplier = loss_multiplier.to(device=advantages.device, dtype=advantages.dtype)
                    # Ensure this is used as a constant scaling factor only.
                    sample_loss_multiplier = sample_loss_multiplier.detach().view(-1)

                    entropy_coeff = self.config.entropy_coeff

                    # Each group loss below is scaled into its exact contribution
                    # to the complete mini-batch objective.
                    loss_scale_factor = 1.0

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Build per-row groups, then compute each group's expected loss
                    # separately and average the row-multiplier-weighted group
                    # expectations. Splitting otherwise identical data into multiple
                    # groups therefore does not change the total actor-loss scale.
                    #
                    # Grouping rules:
                    #   * When `loss_group_id` is present, group strictly by it
                    #     (set by the trainer to e.g. "main_rollout" / "sub_rollout").
                    #     `loss_multiplier` remains a per-row weight within the group.
                    #   * When `loss_group_id` is absent, fall back to grouping by
                    #     `loss_multiplier` value (preserves flows such as noise_learn
                    #     that only tag rows via multiplier).
                    loss_group_id = model_inputs.get("loss_group_id", None)
                    group_keys = self._loss_group_keys(loss_group_id, sample_loss_multiplier)

                    # group key -> list of row indices
                    groups: dict[object, list[int]] = {}
                    for i, key in enumerate(group_keys):
                        groups.setdefault(key, []).append(i)

                    # Use a full-batch normalizer when supplied. Otherwise the number
                    # of active groups in the complete mini-batch is the stable fallback;
                    # using the per-micro-batch group count would make the loss depend on
                    # how dynamic batching happened to partition the rows.
                    global_normalizer_field = model_inputs.get("loss_group_normalizer", None)
                    if global_normalizer_field is not None:
                        if not torch.is_tensor(global_normalizer_field):
                            global_normalizer_field = torch.as_tensor(
                                global_normalizer_field,
                                device=advantages.device,
                                dtype=advantages.dtype,
                            )
                        else:
                            global_normalizer_field = global_normalizer_field.to(
                                device=advantages.device,
                                dtype=advantages.dtype,
                            )
                        normalizer_val = float(global_normalizer_field.detach().view(-1)[0].item())
                        loss_group_normalizer = (
                            normalizer_val
                            if normalizer_val > 0
                            else float(max(mini_active_group_count, 1))
                        )
                    else:
                        loss_group_normalizer = float(max(mini_active_group_count, 1))

                    def _grouped_agg_loss(loss_mat: torch.Tensor) -> torch.Tensor:
                        """Compose group aggregates into the complete mini-batch objective."""
                        total = torch.zeros((), device=loss_mat.device, dtype=loss_mat.dtype)
                        for group_key, idx_list in groups.items():
                            if not idx_list:
                                continue
                            g_idx = torch.as_tensor(idx_list, device=loss_mat.device, dtype=torch.long)
                            g_multiplier = sample_loss_multiplier.index_select(0, g_idx).to(
                                device=loss_mat.device,
                                dtype=loss_mat.dtype,
                            ).view(-1, 1)
                            group_val = agg_loss(
                                loss_mat=loss_mat[g_idx] * g_multiplier,
                                loss_mask=response_mask[g_idx],
                                loss_agg_mode=loss_agg_mode,
                            )
                            contribution_scale = self._micro_batch_aggregation_scale(
                                response_mask[g_idx],
                                global_mass=global_group_masses.get(group_key, 0.0),
                                loss_agg_mode=loss_agg_mode,
                            )
                            total = total + group_val * contribution_scale
                        return total / loss_group_normalizer

                    pg_loss = torch.zeros((), device=advantages.device, dtype=advantages.dtype)
                    # Aggregate per-group pg metrics (pg_clipfrac, ppo_kl, pg_clipfrac_lower)
                    # by token-count weighting so the logged values still equal the global
                    # batch-level metric (keeps dashboards comparable across the change).
                    pg_metrics_acc: dict[str, list[float]] = {}
                    for group_key, idx_list in groups.items():
                        if not idx_list:
                            continue
                        g_idx = torch.as_tensor(idx_list, device=advantages.device, dtype=torch.long)
                        g_response_mask = response_mask[g_idx]
                        g_multiplier = sample_loss_multiplier.index_select(0, g_idx).to(
                            device=advantages.device,
                            dtype=advantages.dtype,
                        ).view(-1, 1)
                        group_pg_loss, group_pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob[g_idx],
                            log_prob=log_prob[g_idx],
                            advantages=advantages[g_idx] * g_multiplier,
                            response_mask=g_response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=(
                                rollout_is_weights[g_idx] if rollout_is_weights is not None else None
                            ),
                        )
                        contribution_scale = self._micro_batch_aggregation_scale(
                            g_response_mask,
                            global_mass=global_policy_group_masses.get(group_key, 0.0),
                            loss_agg_mode=policy_loss_agg_mode,
                        )
                        pg_loss = pg_loss + group_pg_loss * contribution_scale
                        g_weight = float(g_response_mask.sum().detach().item())
                        for k, v in group_pg_metrics.items():
                            cum_sum, cum_w = pg_metrics_acc.get(k, [0.0, 0.0])
                            pg_metrics_acc[k] = [cum_sum + float(v) * g_weight, cum_w + g_weight]
                    pg_loss = pg_loss / loss_group_normalizer
                    pg_metrics = {
                        k: (s / w if w > 0 else 0.0) for k, (s, w) in pg_metrics_acc.items()
                    }
                    micro_batch_metrics.update(pg_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = _grouped_agg_loss(entropy)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = _grouped_agg_loss(kld)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
