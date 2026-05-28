# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Ray single-controller trainer for ``denoise`` (extends VERL ``RayPPOTrainer``).

Per ``train_batch`` we issue **one** ``generate_sequences`` call that produces ``N + K``
trajectories per prompt:

* ``N`` "main" rollouts: standard generation from the original prompt.
* ``K`` "sub" rollouts: continuation from a partial wrong solution drawn from the
  ``wrong_answer_with_boxed`` column (produced by ``data_prepare.py``). The first
  ``part_response_ratio`` of the wrong solution (measured in **tokens**, default 0.5)
  is appended to the prompt as an ``assistant`` message with
  ``continue_final_message=True`` so the policy continues writing from that prefix.

For sub-rollouts, the partial-wrong prefix is folded into the response window after
generation. ``trainer.partial_mode`` selects one of three behaviors:

* ``"shift"`` (default): response width = ``R``; trailing ``p_i`` rollout tokens are
  discarded so ``partial_wrong + kept_continuation <= R`` (length-fair vs main
  rollouts); partial-wrong gets ``response_mask = 1`` so PPO loss covers
  ``[partial_wrong, kept_continuation]``. More signal, but gradients flow through
  off-policy prefix tokens, which can be unstable when the prefix is heavily
  off-policy.
* ``"cutdown"``: response width = ``R``; same length-fair truncation as ``"shift"``,
  but partial-wrong gets ``response_mask = 0`` so PPO loss only covers
  ``kept_continuation``. More stable, less signal.
* ``"none"``: response width = ``R + max_partial_len``; NO truncation — total per-row
  output is ``p_i + R`` (not length-fair vs main rollouts but preserves every
  generated token). Partial-wrong gets ``response_mask = 0`` (no gradient on the
  off-policy prefix); all ``R`` generated tokens participate in PPO loss.

In every mode the partial-wrong prefix remains visible to the reward manager via
``attention_mask`` so the decoded answer contains the full
``[partial_wrong, continuation]`` sequence (decoded length is ``p_i + kept`` for
``shift`` / ``cutdown`` and ``p_i + R`` for ``none``). Rows without a partial prefix
(the ``N`` main rollouts) are unaffected by the mode.
"""

import json
import math
import os
import uuid
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict
from tqdm import tqdm

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
def ensure_problem_ids_on_dataset(dataset) -> None:
    """If the dataset has no ``problem_id`` column, assign 1..N per row."""
    if dataset is None or not hasattr(dataset, "dataframe"):
        return
    df = dataset.dataframe
    n = len(df)
    if n == 0:
        return
    if "problem_id" not in df.column_names:
        dataset.dataframe = df.add_column("problem_id", [i + 1 for i in range(n)])
        print(f"[denoise] dataset missing 'problem_id'; assigned 1..{n}.")


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------
def compute_problem_id_to_avg_acc(batch: DataProto) -> Tuple[dict, dict]:
    """
    Aggregate verifier ``acc`` by ``problem_id`` over rollouts.

    Returns:
        (problem_id -> mean acc, distribution metrics dict).
    """
    acc_vals = batch.non_tensor_batch.get("acc", None)
    problem_ids = batch.non_tensor_batch.get("problem_id", None)
    if acc_vals is None or problem_ids is None:
        raise ValueError(f"acc is {acc_vals}, problem_id is {problem_ids}.")
    problem_ids = np.asarray(problem_ids)
    if problem_ids.size == 0:
        raise ValueError("len(problem_ids) == 0.")

    acc_vals = np.asarray(acc_vals, dtype=np.float32)
    pid_to_acc_list: dict = {}
    n = min(len(problem_ids), len(acc_vals))
    for i in range(n):
        pid = problem_ids[i]
        pid_to_acc_list.setdefault(pid, []).append(float(acc_vals[i]))

    problem_id_to_avg_acc = {pid: sum(vals) / len(vals) for pid, vals in pid_to_acc_list.items()}
    return problem_id_to_avg_acc, problem_id_avg_acc_distribution_metrics(problem_id_to_avg_acc)


def problem_id_avg_acc_distribution_metrics(problem_id_to_avg_acc: dict) -> dict:
    """Discrete distribution metrics over per-problem mean acc."""
    if not problem_id_to_avg_acc:
        return {}

    tol = 1e-5
    n_eq_0 = n_eq_1 = n_open_0_05 = n_open_05_1 = 0

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= tol

    for a in problem_id_to_avg_acc.values():
        if not np.isfinite(a):
            continue
        if close(a, 0.0):
            n_eq_0 += 1
        elif close(a, 1.0):
            n_eq_1 += 1
        elif close(a, 0.5):
            pass
        elif 0.0 < a < 0.5:
            n_open_0_05 += 1
        elif 0.5 < a < 1.0:
            n_open_05_1 += 1

    n = len([a for a in problem_id_to_avg_acc.values() if np.isfinite(a)])
    out = {
        "denoise/problem_id_avg_acc/n_problems": float(n),
        "denoise/problem_id_avg_acc/n_eq_0": float(n_eq_0),
        "denoise/problem_id_avg_acc/n_eq_1": float(n_eq_1),
        "denoise/problem_id_avg_acc/n_in_open_0_0.5": float(n_open_0_05),
        "denoise/problem_id_avg_acc/n_in_open_0.5_1": float(n_open_05_1),
    }
    if n > 0:
        finite = [float(a) for a in problem_id_to_avg_acc.values() if np.isfinite(a)]
        out["denoise/problem_id_avg_acc/mean"] = float(np.mean(finite))
    return out


def decode_rollout_response_str(tokenizer, prompt_ids, response_ids, attention_mask):
    """
    Decode the valid response segment for a single rollout row. Layout matches VERL after
    union(generate): ``attention_mask`` is ``[prompt_part | response_part]``; prompts are
    left-padded, responses are right-padded.

    Returns:
        (decoded_response_str, valid_response_length).
    """
    prompt_length = prompt_ids.shape[-1]
    valid_response_length = int(attention_mask[prompt_length:].sum().item())
    valid_response_ids = response_ids[:valid_response_length]
    if isinstance(valid_response_ids, torch.Tensor):
        valid_response_ids = valid_response_ids.detach().cpu()
    text = tokenizer.decode(valid_response_ids, skip_special_tokens=True).strip()
    return text, valid_response_length


def save_rollout_with_token_level_reward_sum(
    batch: DataProto,
    tokenizer,
    rollout_save_path: Optional[str],
) -> None:
    """Save rollout rows with score = sum(token_level_rewards over valid response tokens).

    Splits records into two files based on whether the row has a partial wrong prefix
    (``partial_response_len > 0`` -> ``sub_rollout.jsonl``; otherwise ``rollout_save_path``).
    """
    if batch is None or not rollout_save_path:
        return
    if "token_level_rewards" not in batch.batch.keys():
        return

    def _to_jsonable(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(v) for v in value]
        return value

    rollout_path = Path(rollout_save_path)
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    sub_path = rollout_path.parent / "sub_rollout.jsonl"

    file_handles: dict = {}

    def _get_handle(path: Path):
        key = str(path.resolve())
        if key not in file_handles:
            file_handles[key] = path.open("a", encoding="utf-8")
        return file_handles[key]

    try:
        partial_lens = batch.non_tensor_batch.get("partial_response_len", None)
        for i in range(len(batch)):
            data_item = batch[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str, valid_response_length = decode_rollout_response_str(
                tokenizer, prompt_ids, response_ids, data_item.batch["attention_mask"]
            )

            token_level_rewards = data_item.batch["token_level_rewards"]
            score = float(token_level_rewards[:valid_response_length].sum().item())
            reward_model = data_item.non_tensor_batch.get("reward_model", {})
            ground_truth = reward_model.get("ground_truth", None) if reward_model else None
            p_len = int(partial_lens[i]) if partial_lens is not None else 0

            rollout_record = {
                "prompt": prompt_str,
                "response": response_str,
                "ground_truth": _to_jsonable(ground_truth),
                "score": _to_jsonable(score),
                "partial_response_len": p_len,
            }
            target_path = sub_path if p_len > 0 else rollout_path
            _get_handle(target_path).write(json.dumps(rollout_record, ensure_ascii=False) + "\n")
    finally:
        for file_handle in file_handles.values():
            file_handle.close()


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class RayDAPOTrainer(RayPPOTrainer):
    """
    DAPO-style trainer with ``N + K`` unified rollouts per problem.

    The ``K`` sub-rollouts are generated from a partial wrong solution (first
    ``part_response_ratio`` of a row drawn from ``wrong_answer_with_boxed``) appended to
    the prompt as an assistant message. After rollout the partial-wrong prefix is folded
    into the response window per ``trainer.partial_mode``:

    * ``"shift"``   — response width = ``R``, length-fair truncation
      (``partial_wrong + kept <= R``), partial-wrong ``response_mask = 1``
      (gradient on the prefix).
    * ``"cutdown"`` — response width = ``R``, same truncation as ``"shift"``,
      partial-wrong ``response_mask = 0`` (no gradient on the prefix).
    * ``"none"``    — response width = ``R + max_partial_len``, NO truncation
      (per-row output is ``p_i + R``), partial-wrong ``response_mask = 0``.

    See module docstring for details.
    """

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        """
        Recompute ``old_log_prob`` (+ ref if enabled), entropy metric, and union onto ``batch``.

        ``response_mask`` is only (re)computed here if not already set: sub-rollouts have a
        custom mask attached in ``_rollout_and_compute_reward`` that we must not overwrite.
        """
        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            metrics["actor/entropy"] = entropy_agg.detach().item()
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, "olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def fit(self):
        """Main training loop: each step runs ``train_batch`` (rollout + reward + PPO update)."""
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.tracking_logger = logger

        self.global_steps = 0
        self.inner_global_steps = 0
        self.gen_steps = 0

        self._load_checkpoint()

        ensure_problem_ids_on_dataset(self.train_dataset)

        # Build problem_id -> raw item mapping (used to fetch wrong_answer_with_boxed and prompt
        # messages for sub-rollouts; the trainer-side data layout is preserved unchanged).
        self.all_train_items = {}
        for item in self.train_dataset.dataframe:
            self.all_train_items[item["problem_id"]] = item

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics:{val_metrics}")
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        self.inner_global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None

        # DAPO dynamic-sampling toggle and accumulation state.
        # When enabled, each ``batch_dict`` yields a ``data.gen_batch_size`` chunk that
        # is rolled out and filtered (drop problems whose avg acc == 0 or 1). Kept
        # problems are concatenated across gen_batches until we have
        # ``data.train_batch_size`` problems, then a single PPO update runs on the
        # first ``train_batch_size`` problems.
        use_dapo = bool(self.config.trainer.get("use_dapo", False))
        dapo_state: dict = {
            "batch": None,
            "num_kept_problems": 0,
            "num_gen_batches": 0,
            "metrics": {},
            "reward_extra_infos_dict": {},
        }

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if self.inner_global_steps < self.global_steps:
                    self.inner_global_steps += 1
                    continue
                is_last_step = self.global_steps >= self.total_training_steps
                if use_dapo:
                    result = self._dapo_step(
                        batch_dict,
                        prev_step_profile,
                        curr_step_profile,
                        timing_raw,
                        dapo_state,
                    )
                    if result is None:
                        # Not enough kept problems yet; consume another gen_batch
                        # without advancing the global step.
                        self.gen_steps += 1
                        self.inner_global_steps += 1
                        continue
                    batch, metrics = result
                else:
                    batch, metrics = self.train_batch(
                        batch_dict,
                        prev_step_profile,
                        curr_step_profile,
                        timing_raw,
                    )
                problem_id_to_avg_acc, problem_id_metrics = compute_problem_id_to_avg_acc(batch)
                if self.global_steps % self.config.trainer.save_freq == 1:
                    local_global_step_save_json = os.path.join(
                        self.config.trainer.default_local_dir,
                        f"global_step_{self.global_steps}/rollout.jsonl",
                    )
                    save_rollout_with_token_level_reward_sum(
                        batch=batch,
                        tokenizer=self.tokenizer,
                        rollout_save_path=local_global_step_save_json,
                    )
                print(
                    "has_none_pid:", None in problem_id_to_avg_acc,
                    "none_pid_avg:", problem_id_to_avg_acc.get(None),
                )
                metrics.update(problem_id_metrics)
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)

                if "acc" in batch.non_tensor_batch:
                    acc_vals = np.asarray(batch.non_tensor_batch["acc"], dtype=np.float32)
                    acc_finite = acc_vals[np.isfinite(acc_vals)]
                    if acc_finite.size > 0:
                        metrics["reward_model/acc"] = float(np.mean(acc_finite))

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics:{last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.inner_global_steps += 1
                self.gen_steps += 1

                batch = None

        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)

    # -------------------------------------------------------------------------
    # Sub-rollout helpers
    # -------------------------------------------------------------------------
    def _make_part_response_ratio_sampler(self):
        """Build a zero-arg callable that returns the next ``part_response_ratio``.

        The strategy is selected by ``trainer.part_response_ratio_strategy``:
          * ``"fixed"``   - constant ratio (``part_response_ratio_fixed`` if set, else
                            falls back to the legacy ``part_response_ratio`` field so
                            older scripts keep working).
          * ``"normal"``  - draw from ``N(mean, std)`` then clip to
                            ``[low, high]``.
          * ``"uniform"`` - draw uniformly from ``[low, high]``.

        Bounds must satisfy ``0 < low <= high <= 1``. The returned ratio is always in
        ``(0, 1]`` so downstream tokenization invariants (``cut >= 1``) hold.
        """
        cfg = self.config.trainer
        strategy = str(cfg.get("part_response_ratio_strategy", "fixed")).lower()

        def _coerce_ratio(name: str, value, *, allow_one: bool = True) -> float:
            f = float(value)
            upper_ok = (f <= 1.0) if allow_one else (f < 1.0)
            if not (0.0 < f and upper_ok):
                bound = "(0, 1]" if allow_one else "(0, 1)"
                raise ValueError(f"trainer.{name} must be in {bound}, got {f}.")
            return f

        if strategy == "fixed":
            raw = cfg.get("part_response_ratio_fixed", 0.2)
            ratio = _coerce_ratio("part_response_ratio_fixed", raw)

            def _sample_fixed() -> float:
                return ratio

            return _sample_fixed

        if strategy not in ("normal", "uniform"):
            raise ValueError(
                "trainer.part_response_ratio_strategy must be one of "
                f"'fixed' | 'normal' | 'uniform', got {strategy!r}."
            )

        low = _coerce_ratio("part_response_ratio_low", cfg.get("part_response_ratio_low", 0.1))
        high = _coerce_ratio("part_response_ratio_high", cfg.get("part_response_ratio_high", 0.9))
        if low > high:
            raise ValueError(
                f"trainer.part_response_ratio_low ({low}) must be <= "
                f"trainer.part_response_ratio_high ({high})."
            )

        rng = np.random.default_rng()

        if strategy == "uniform":
            def _sample_uniform() -> float:
                return float(rng.uniform(low, high))

            return _sample_uniform

        # strategy == "normal"
        mean = _coerce_ratio("part_response_ratio_mean", cfg.get("part_response_ratio_mean", 0.5))
        std = float(cfg.get("part_response_ratio_std", 0.2))
        if std < 0.0:
            raise ValueError(f"trainer.part_response_ratio_std must be >= 0, got {std}.")

        def _sample_normal() -> float:
            x = float(rng.normal(mean, std)) if std > 0.0 else mean
            if x < low:
                x = low
            elif x > high:
                x = high
            return x

        return _sample_normal

    def _select_wrong_solutions(self, problem_id, k: int) -> List[str]:
        """Pick up to ``k`` wrong solutions for ``problem_id`` (cycling/replacement when needed)."""
        item = self.all_train_items.get(problem_id)
        if item is None:
            return []
        wrongs = item.get("wrong_answer_with_boxed", None)
        if wrongs is None:
            return []
        # ``wrong_answer_with_boxed`` may come back as a numpy/Arrow sequence; coerce to list[str].
        try:
            wrongs = list(wrongs)
        except TypeError:
            return []
        wrongs = [w for w in wrongs if isinstance(w, str) and w.strip()]

        wrongs = wrongs[:1]
        if not wrongs or k <= 0:
            return []
        # Use a deterministic ordering plus cycle/sample when there are fewer than k candidates.
        if len(wrongs) >= k:
            return wrongs[:k]
        # Cycle through the available wrongs to reach exactly k items.
        return [wrongs[j % len(wrongs)] for j in range(k)]

    def _build_partial_inputs(
        self,
        prompt_messages: List[dict],
        wrong_text: str,
        part_response_ratio: float,
        max_prompt_length: int,
        truncation: str,
        pad_token_id: int,
        apply_kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, int]:
        """
        Render a prompt where the assistant message is the first ``part_response_ratio``
        (by tokens) of ``wrong_text``. The rendering uses ``continue_final_message=True``
        so the model continues writing directly from the partial prefix.

        Returns:
            (input_ids[1, P], attention_mask[1, P], position_ids[1, P], raw_prompt_ids, partial_token_len)
            where ``P = max_prompt_length`` (left-padded).
        """
        # Cut the wrong solution to the first part_response_ratio of its tokens.
        wrong_token_ids = self.tokenizer.encode(wrong_text, add_special_tokens=False)
        if not wrong_token_ids:
            raise ValueError("Empty wrong solution after tokenization.")
        cut = int(len(wrong_token_ids) * part_response_ratio)
        cut = max(1, min(cut, len(wrong_token_ids)))
        partial_text = self.tokenizer.decode(wrong_token_ids[:cut], skip_special_tokens=True)

        # Render base prefix ([sys, user] with assistant generation prompt). Strip any
        # incoming continue_final_message flag so the base prefix is rendered cleanly.
        clean_kwargs = {k: v for k, v in apply_kwargs.items() if k != "continue_final_message"}
        base_text = self.tokenizer.apply_chat_template(
            list(prompt_messages),
            add_generation_prompt=True,
            tokenize=False,
            **clean_kwargs,
        )
        full_text = base_text + partial_text

        # Compute partial token length in the **rendered** sequence.
        base_ids = self.tokenizer.encode(base_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        partial_token_len = max(0, len(full_ids) - len(base_ids))

        # Tokenize the full prompt for model input.
        model_inputs = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        ids = model_inputs["input_ids"]
        mask = model_inputs["attention_mask"]
        ids, mask = verl_F.postprocess_data(
            input_ids=ids,
            attention_mask=mask,
            max_length=max_prompt_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation=truncation,
        )
        pos = compute_position_id_with_mask(mask)

        raw_prompt_ids = list(full_ids)
        if len(raw_prompt_ids) > max_prompt_length:
            if truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]
            elif truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:max_prompt_length]
            elif truncation == "middle":
                left_half = max_prompt_length // 2
                right_half = max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif truncation == "error":
                raise RuntimeError(
                    f"Partial prompt length {len(raw_prompt_ids)} > max_prompt_length {max_prompt_length}."
                )

        # If post-padding truncation dropped some of the partial prefix tokens, adjust to keep
        # ``partial_token_len`` consistent with the final tokenized window.
        actual_len = int(mask.sum().item())
        partial_token_len = min(partial_token_len, max(0, actual_len - 1))

        return ids, mask, pos, raw_prompt_ids, partial_token_len

    def _fold_partial_into_response(
        self,
        batch: DataProto,
        partial_lens: np.ndarray,
        max_partial_len: int,
        mask_partial_wrong: bool,
    ) -> DataProto:
        """
        Per-row right-shift of the partial-wrong prefix into the response window,
        keeping the response width fixed at ``R``. This enforces a fair output budget:
        ``partial_wrong_len + kept_continuation_len <= R`` independent of ``p_i``;
        the trailing ``p_i`` rollout tokens are silently discarded.

        Two modes share this transform and differ only in whether the partial-wrong
        region of ``response_mask`` is set to ``1`` (gradient flows through the off-
        policy prefix) or ``0`` (no gradient on the prefix):

        * ``mask_partial_wrong = False`` (``"shift"``): partial-wrong gets ``mask = 1``;
          PPO loss covers ``[partial_wrong, kept_continuation]``. More signal but
          potentially less stable when the prefix is heavily off-policy.
        * ``mask_partial_wrong = True``  (``"cutdown"``): partial-wrong gets ``mask = 0``;
          PPO loss only covers ``kept_continuation``. More stable, less signal.

        For each row ``i`` with ``p_i = partial_lens[i] in [0, max_partial_len]`` we
        right-shift the full ``(BS, P + R)`` matrix by ``p_i`` columns (filling the
        leading ``p_i`` positions with PAD / 0). This:

        * moves the rightmost ``p_i`` prompt tokens (the partial-wrong prefix) to the
          beginning of the response window, and
        * drops the trailing ``p_i`` tokens of the rollout — these are the last ``p_i``
          generated tokens (or, when the rollout terminated early, just trailing
          response padding).

        Shapes are preserved: ``prompts`` is still ``(BS, P)`` and ``responses`` is
        still ``(BS, R)``. The reward manager's standard
        ``valid_response_length = attention_mask[:, prompt_length:].sum()`` returns
        ``p_i + kept_response_length`` for sub-rollouts, so the decoded response slice
        contains exactly ``[partial_wrong, kept_continuation]`` (and only the generated
        tokens for main rollouts; no chat-template leakage). Only ``response_mask``
        differs between the two modes — the input tensor layout is identical.

        Specifically:

        * ``new_prompts[i]`` has the same ``P`` columns, but with ``p_i`` extra leading
          PADs (the real prompt content has shrunk by ``p_i`` because partial_wrong
          moved into the response). For main rollouts (``p_i = 0``) prompt content is
          preserved exactly.
        * ``new_responses[i] = [partial_wrong (p_i), generated[:R - p_i]]`` — exactly
          ``R`` tokens wide. No trailing PADs are added beyond what the rollout
          already had.
        * ``response_mask[i]`` is ``1`` (shift) or ``0`` (cutdown) over the leading
          ``p_i`` partial-wrong positions, and over ``[p_i, R)`` follows either a
          rollout-provided inner mask (e.g. sglang multi-turn loss masking) or the
          new attention mask, both right-truncated to ``R - p_i`` columns.

        Tokens past the budget (the last ``p_i`` rollout tokens) are silently
        discarded; the caller surfaces a metric on how many were real (non-pad)
        generated tokens so the cost of the cap is visible. Requires
        ``max_partial_len <= R``.

        ``rollout_log_probs`` (shape ``(BS, R)``), if present, is also right-shifted
        per row so that row ``i``'s first ``R - p_i`` original log probs land at
        positions ``[p_i, R)`` of the new response window with zeros at ``[0, p_i)``;
        the last ``p_i`` entries are dropped along with the truncated tokens.
        Downstream rollout-correction metrics consume this masked by response_mask,
        so for cutdown the zero-padded prefix is naturally ignored, while for shift
        the IS-weight diagnostic at those tokens is approximate (logged as zero).
        """
        input_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"]
        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]

        BS, P = prompts.shape
        R = responses.shape[1]
        T = P + R

        if max_partial_len > R:
            raise ValueError(
                f"max_partial_len ({max_partial_len}) exceeds the response length ({R}). "
                f"Increase data.max_response_length or reduce trainer.part_response_ratio."
            )
        if input_ids.shape != (BS, T) or attention_mask.shape != (BS, T):
            raise ValueError(
                f"Inconsistent shapes: input_ids={tuple(input_ids.shape)}, "
                f"attention_mask={tuple(attention_mask.shape)}, expected ({BS}, {T})."
            )

        device = input_ids.device
        pad_token_id = int(self.tokenizer.pad_token_id)

        partial_lens_t = torch.from_numpy(
            np.asarray(partial_lens, dtype=np.int64)
        ).to(device)  # (BS,)

        # Per-row right-shift by p_i: new[i, j] = old[i, j - p_i] for j >= p_i, else PAD/0.
        # The last p_i tokens of old are dropped (trailing rollout tokens, including
        # real generated tokens past the R-budget once partial_wrong is accounted for).
        positions_T = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        plens_col = partial_lens_t.unsqueeze(1)  # (BS, 1)
        src_idx_T = positions_T - plens_col  # (BS, T)
        inbound_T = src_idx_T >= 0  # True for j >= p_i; the leading p_i positions are new PADs
        safe_src_idx_T = torch.where(inbound_T, src_idx_T, torch.zeros_like(src_idx_T))

        gathered_ids = torch.gather(input_ids, dim=1, index=safe_src_idx_T)
        new_input_ids = torch.where(
            inbound_T, gathered_ids, torch.full_like(gathered_ids, pad_token_id)
        ).contiguous()

        gathered_mask = torch.gather(attention_mask, dim=1, index=safe_src_idx_T)
        new_attention_mask = torch.where(
            inbound_T, gathered_mask, torch.zeros_like(gathered_mask)
        ).contiguous()

        # Position ids recomputed from the new attention mask. We only support 2D position_ids
        # in this text-only flow; the upstream rollout asserts that (no multi_modal_data).
        new_position_ids = compute_position_id_with_mask(new_attention_mask).contiguous()

        new_prompts = new_input_ids[:, :P].contiguous()
        new_responses = new_input_ids[:, P:].contiguous()

        # Build the new response_mask over the (R,) window:
        #   * positions [0, p_i): partial-wrong region. mask = 1 in shift mode (gradient
        #     flows through the off-policy prefix), or 0 in cutdown mode (no gradient).
        #   * positions [p_i, R): kept-continuation region. mask follows the rollout-
        #     provided response_mask if available (e.g. sglang multi-turn loss masking),
        #     otherwise the new attention mask over the response window. Both are
        #     right-truncated to R - p_i columns by the gather + inner_region masking
        #     below (the last p_i mask positions are dropped along with the tokens).
        r_positions = torch.arange(R, device=device).unsqueeze(0)  # (1, R)
        src_idx_R = r_positions - plens_col  # (BS, R)
        inner_region = src_idx_R >= 0  # True for [p_i, R)

        if (
            "response_mask" in batch.batch.keys()
            and batch.batch["response_mask"].shape[1] == R
        ):
            inner_mask_R = batch.batch["response_mask"]  # (BS, R)
        else:
            inner_mask_R = attention_mask[:, -R:]  # original attention over the generated window

        safe_src_idx_R = torch.where(inner_region, src_idx_R, torch.zeros_like(src_idx_R))
        inner_gathered = torch.gather(inner_mask_R, dim=1, index=safe_src_idx_R)
        inner_full = torch.where(
            inner_region, inner_gathered, torch.zeros_like(inner_gathered)
        )
        if mask_partial_wrong:
            # cutdown: partial-wrong region stays at 0 (no gradient on the prefix).
            new_response_mask = inner_full.contiguous()
        else:
            # shift: partial-wrong region set to 1 (gradient flows through the prefix).
            partial_region = ~inner_region  # True for [0, p_i)
            new_response_mask = torch.where(
                partial_region, torch.ones_like(inner_full), inner_full
            ).contiguous()

        batch.batch["input_ids"] = new_input_ids
        batch.batch["attention_mask"] = new_attention_mask
        batch.batch["position_ids"] = new_position_ids
        batch.batch["prompts"] = new_prompts
        batch.batch["responses"] = new_responses
        batch.batch["response_mask"] = new_response_mask

        # Per-row right-shift of rollout_log_probs (shape (BS, R)) to align with the
        # new response window: row i's first R - p_i original log probs land at
        # positions [p_i, R); positions [0, p_i) are zeroed (partial-wrong region),
        # and the last p_i entries are dropped with the truncated response tokens.
        if "rollout_log_probs" in batch.batch.keys():
            rlp = batch.batch["rollout_log_probs"]
            if rlp.shape[1] == R:
                gathered_rlp = torch.gather(rlp, dim=1, index=safe_src_idx_R)
                new_rlp = torch.where(
                    inner_region, gathered_rlp, torch.zeros_like(gathered_rlp)
                ).contiguous()
                batch.batch["rollout_log_probs"] = new_rlp

        return batch

    def _fold_partial_extended(
        self,
        batch: DataProto,
        partial_lens: np.ndarray,
        max_partial_len: int,
    ) -> DataProto:
        """
        Per-row LEFT-shift that folds the partial-wrong prefix into the response
        without any length truncation: the response window is EXTENDED to width
        ``R + max_partial_len`` so the full ``p_i`` prefix and all ``R`` generated
        tokens are preserved. Used for ``trainer.partial_mode = "none"``.

        Total per-row output length is ``p_i + R`` for sub-rollouts (and ``R`` for
        main rollouts), so sub-rollouts effectively get up to ``max_partial_len``
        extra tokens of output budget compared to main rollouts — i.e. NO fairness
        cap, in contrast to ``_fold_partial_into_response`` which truncates the
        trailing ``p_i`` tokens to enforce ``partial_wrong + kept <= R``.

        For each row ``i`` with ``p_i = partial_lens[i] in [0, max_partial_len]`` we
        left-shift the full ``(BS, P + R)`` matrix by ``offset_i = max_partial_len - p_i``
        columns (filling the trailing ``offset_i`` positions with PAD / 0). After the
        shift:

        * ``new_prompts[i]`` has width ``P - max_partial_len`` and is the original
          prompt with its last ``p_i`` tokens removed (those moved into the response),
          aligned to the common shape by trimming ``offset_i`` leading PADs. For main
          rollouts (``p_i = 0``) prompt content is preserved exactly — only the
          leading PAD count shrinks by ``max_partial_len``.
        * ``new_responses[i]`` has width ``R + max_partial_len`` and equals
          ``[partial_wrong (p_i), generated (R), trailing PADs (max_partial_len - p_i)]``.
          For main rollouts this is the original response with extra trailing PADs.
        * ``response_mask[i]`` is ``0`` over the leading ``p_i`` partial-wrong
          positions (no gradient on the off-policy prefix), then follows either a
          rollout-provided inner mask (e.g. sglang multi-turn loss masking) or the
          new attention mask over the inner ``R`` window, and is ``0`` on the
          trailing PADs.

        The standard VERL reward-manager pattern
        ``valid_response_length = attention_mask[:, prompt_length:].sum()`` returns
        ``p_i + actual_response_length`` for sub-rollouts and ``actual_response_length``
        for main rollouts, so the decoded response slice contains exactly
        ``[partial_wrong, full_continuation]`` for sub-rollouts.

        Requires each row to have at least ``offset_i`` leading PAD tokens in the
        prompt (equivalently, the real prompt content must fit in
        ``P - max_partial_len + p_i`` tokens); a clear error is raised otherwise.

        ``rollout_log_probs`` (shape ``(BS, R)``), if present, is also shifted per
        row so each row's original log probs land at positions ``[p_i, p_i + R)`` of
        the new response window with zeros elsewhere. Downstream rollout-correction
        metrics consume this masked by response_mask, so the zero-padded prefix
        (mask = 0) is naturally ignored.
        """
        input_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"]
        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]

        BS, P = prompts.shape
        R = responses.shape[1]
        T = P + R

        if P < max_partial_len:
            raise ValueError(
                f"max_partial_len ({max_partial_len}) exceeds the prompt length ({P}). "
                f"Increase data.max_prompt_length or reduce trainer.part_response_ratio."
            )
        if input_ids.shape != (BS, T) or attention_mask.shape != (BS, T):
            raise ValueError(
                f"Inconsistent shapes: input_ids={tuple(input_ids.shape)}, "
                f"attention_mask={tuple(attention_mask.shape)}, expected ({BS}, {T})."
            )

        device = input_ids.device
        pad_token_id = int(self.tokenizer.pad_token_id)

        partial_lens_t = torch.from_numpy(
            np.asarray(partial_lens, dtype=np.int64)
        ).to(device)  # (BS,)
        offsets_t = (max_partial_len - partial_lens_t).to(torch.long)  # (BS,) in [0, max_partial_len]

        # Safety check: row i needs >= offset_i leading PADs in its prompt; otherwise
        # the per-row left-shift would silently drop real prompt tokens off the front.
        inspect_pos = torch.arange(max_partial_len, device=device).unsqueeze(0)  # (1, max_partial_len)
        inspect_mask = (inspect_pos < offsets_t.unsqueeze(1)).to(attention_mask.dtype)
        real_in_left_strip = (attention_mask[:, :max_partial_len] * inspect_mask).sum(dim=-1)
        if torch.any(real_in_left_strip > 0):
            n_bad = int((real_in_left_strip > 0).sum().item())
            max_overflow = int(real_in_left_strip.max().item())
            raise ValueError(
                f"_fold_partial_extended: {n_bad} row(s) have real prompt tokens in "
                f"the leading {max_partial_len} positions that would be dropped by the per-row "
                f"left-shift (worst-case overflow = {max_overflow} tokens). Increase "
                f"data.max_prompt_length so each prompt has at least max_partial_len more "
                f"left-padding (equivalently, real_prompt_len <= P - max_partial_len + p_i)."
            )

        # Per-row left-shift of (input_ids, attention_mask) by offset_i, right-padded
        # with pad_token_id / 0. After this, the new prompt = first (P - max_partial_len)
        # tokens and the new response = last (R + max_partial_len) tokens.
        positions_T = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        gather_idx = positions_T + offsets_t.unsqueeze(1)  # (BS, T)
        inbound = gather_idx < T  # (BS, T) bool
        safe_idx = torch.where(inbound, gather_idx, torch.zeros_like(gather_idx))

        gathered_ids = torch.gather(input_ids, dim=1, index=safe_idx)
        new_input_ids = torch.where(
            inbound, gathered_ids, torch.full_like(gathered_ids, pad_token_id)
        ).contiguous()

        gathered_mask = torch.gather(attention_mask, dim=1, index=safe_idx)
        new_attention_mask = torch.where(
            inbound, gathered_mask, torch.zeros_like(gathered_mask)
        ).contiguous()

        new_position_ids = compute_position_id_with_mask(new_attention_mask).contiguous()

        new_prompt_len = P - max_partial_len
        new_response_len = R + max_partial_len

        new_prompts = new_input_ids[:, :new_prompt_len].contiguous()
        new_responses = new_input_ids[:, new_prompt_len:].contiguous()

        # Build response_mask over the (R + max_partial_len,) window:
        #   * positions [0, p_i):           partial-wrong region, mask = 0 (no gradient)
        #   * positions [p_i, p_i + R):     inner response, mask follows rollout-provided
        #                                   response_mask if available, else attention_mask
        #                                   over the original generated window.
        #   * positions [p_i + R, ..):      trailing PADs introduced by the shift, mask = 0
        rr_positions = torch.arange(new_response_len, device=device).unsqueeze(0)  # (1, RR)
        plens = partial_lens_t.unsqueeze(1)  # (BS, 1)
        src_idx_rr = rr_positions - plens  # (BS, RR)
        inner_region = (src_idx_rr >= 0) & (src_idx_rr < R)  # True for [p_i, p_i + R)

        if (
            "response_mask" in batch.batch.keys()
            and batch.batch["response_mask"].shape[1] == R
        ):
            inner_mask_R = batch.batch["response_mask"]  # (BS, R)
        else:
            inner_mask_R = attention_mask[:, -R:]  # original attention over the generated window

        safe_src_idx_rr = torch.where(inner_region, src_idx_rr, torch.zeros_like(src_idx_rr))
        inner_gathered = torch.gather(inner_mask_R, dim=1, index=safe_src_idx_rr)
        new_response_mask = torch.where(
            inner_region, inner_gathered, torch.zeros_like(inner_gathered)
        ).contiguous()

        batch.batch["input_ids"] = new_input_ids
        batch.batch["attention_mask"] = new_attention_mask
        batch.batch["position_ids"] = new_position_ids
        batch.batch["prompts"] = new_prompts
        batch.batch["responses"] = new_responses
        batch.batch["response_mask"] = new_response_mask

        # rollout_log_probs (shape (BS, R)): left-shift per row so each row's original
        # log probs land at positions [p_i, p_i + R); leading [0, p_i) (partial-wrong)
        # and trailing [p_i + R, ..) (PADs) are zeroed.
        if "rollout_log_probs" in batch.batch.keys():
            rlp = batch.batch["rollout_log_probs"]
            if rlp.shape[1] == R:
                gathered_rlp = torch.gather(rlp, dim=1, index=safe_src_idx_rr)
                new_rlp = torch.where(
                    inner_region, gathered_rlp, torch.zeros_like(gathered_rlp)
                ).contiguous()
                batch.batch["rollout_log_probs"] = new_rlp

        return batch

    # -------------------------------------------------------------------------
    # Core: rollout + reward
    # -------------------------------------------------------------------------
    def _rollout_and_compute_reward(self, batch_dict, _metrics, timing_raw):
        """
        One unified rollout pass producing ``N + K`` trajectories per problem in a single
        ``generate_sequences`` call.

        Steps:
            1. Build ``B * (N + K)`` left-padded prompt tensors. The first ``N`` per problem are
               the standard tokenization; the next ``K`` add a partial wrong solution as an
               ``assistant`` message with ``continue_final_message=True``.
            2. Call ``generate_sequences`` once.
            3. For rows with a partial prefix, fold the prefix into the response window so the
               reward manager sees ``[partial_wrong, continuation]``. Behavior is controlled by
               ``trainer.partial_mode``:
                 * ``"shift"``   — response width = ``R``, truncate the trailing ``p_i`` tokens
                   (``partial_wrong + kept <= R``), partial-wrong ``mask = 1``.
                 * ``"cutdown"`` — response width = ``R``, same truncation as ``"shift"``,
                   partial-wrong ``mask = 0``.
                 * ``"none"``    — response width = ``R + max_partial_len``, no truncation
                   (per-row output is ``p_i + R``), partial-wrong ``mask = 0``.
            4. Compute reward via the standard ``compute_reward`` flow.

        ``_metrics`` is the same metrics dict used by ``train_batch``; sub-rollout planning
        counters and per-batch partial length stats are written into it here.
        """
        n = int(self.config.actor_rollout_ref.rollout.n)
        k = int(self.config.trainer.get("sub_rollout_k", 0))
        if k < 0:
            raise ValueError(f"trainer.sub_rollout_k must be >= 0, got {k}.")
        # Sub-rollouts draw a fresh part_response_ratio per slot via this sampler.
        # The sampler validates configuration once and then returns floats in (0, 1].
        sample_part_response_ratio = self._make_part_response_ratio_sampler()
        partial_mode = str(self.config.trainer.get("partial_mode", "shift")).lower()
        if partial_mode not in ("shift", "cutdown", "none"):
            raise ValueError(
                f"trainer.partial_mode must be 'shift', 'cutdown', or 'none', "
                f"got {partial_mode!r}."
            )
        nk = n + k

        new_batch: DataProto = DataProto.from_single_dict(batch_dict)
        B = len(new_batch.batch)

        if "multi_modal_data" in new_batch.non_tensor_batch.keys():
            raise NotImplementedError(
                "multi_modal_data is not supported in denoise (text-only flow)."
            )
        gen_batch = new_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids"],
        )

        base_input_ids = gen_batch.batch["input_ids"]
        base_attention_mask = gen_batch.batch["attention_mask"]
        base_position_ids = gen_batch.batch["position_ids"]
        base_raw_prompt_ids = gen_batch.non_tensor_batch.get("raw_prompt_ids", None)

        data_cfg = self.config.data
        apply_kwargs = data_cfg.get("apply_chat_template_kwargs", {})
        max_prompt_length = int(data_cfg.get("max_prompt_length", 1024))
        truncation = data_cfg.get("truncation", "error")
        pad_token_id = self.tokenizer.pad_token_id

        problem_ids = new_batch.non_tensor_batch.get("problem_id", None)
        if k > 0 and problem_ids is None:
            raise ValueError(
                "sub_rollout_k > 0 requires 'problem_id' in the non-tensor batch "
                "so that wrong_answer_with_boxed can be fetched from self.all_train_items."
            )

        # Build B * (N + K) rows: layout is [P0_n0, ..., P0_n(N-1), P0_k0, ..., P0_k(K-1), P1_n0, ...].
        flat_input_ids_rows: List[torch.Tensor] = []
        flat_attention_mask_rows: List[torch.Tensor] = []
        flat_position_ids_rows: List[torch.Tensor] = []
        flat_raw_prompt_ids: List[list] = []
        partial_lens_flat: List[int] = []

        # Metrics for the sub-rollout planning step.
        n_sub_rows_with_prefix = 0
        n_sub_rows_fallback = 0
        sum_partial_lens = 0
        max_observed_partial = 0
        problems_without_wrongs = 0

        for b in range(B):
            # N main rollouts: reuse the dataloader tokenization.
            for _ in range(n):
                flat_input_ids_rows.append(base_input_ids[b : b + 1])
                flat_attention_mask_rows.append(base_attention_mask[b : b + 1])
                flat_position_ids_rows.append(base_position_ids[b : b + 1])
                flat_raw_prompt_ids.append(
                    list(base_raw_prompt_ids[b]) if base_raw_prompt_ids is not None else []
                )
                partial_lens_flat.append(0)

            if k <= 0:
                continue

            # K sub-rollouts: append a partial wrong solution and re-tokenize with
            # continue_final_message=True. Wrong solutions come from self.all_train_items.
            problem_id = problem_ids[b]
            original_item = self.all_train_items.get(problem_id, {})
            prompt_messages = original_item.get("prompt", None)
            if prompt_messages is None:
                raise ValueError(
                    f"problem_id {problem_id!r} has no 'prompt' field in self.all_train_items."
                )
            # Normalize to a plain list[dict] in case the storage layer returned a numpy/Arrow type.
            try:
                prompt_messages = [dict(m) for m in list(prompt_messages)]
            except Exception as exc:  # pragma: no cover - defensive only
                raise ValueError(
                    f"Cannot coerce prompt for problem_id {problem_id!r} into list[dict]."
                ) from exc

            wrongs = self._select_wrong_solutions(problem_id, k)
            if not wrongs:
                problems_without_wrongs += 1

            for j in range(k):
                wrong_text = wrongs[j] if j < len(wrongs) else None
                if not wrong_text:
                    # Fall back to a standard rollout (no partial prefix) for this slot.
                    flat_input_ids_rows.append(base_input_ids[b : b + 1])
                    flat_attention_mask_rows.append(base_attention_mask[b : b + 1])
                    flat_position_ids_rows.append(base_position_ids[b : b + 1])
                    flat_raw_prompt_ids.append(
                        list(base_raw_prompt_ids[b]) if base_raw_prompt_ids is not None else []
                    )
                    partial_lens_flat.append(0)
                    n_sub_rows_fallback += 1
                    continue

                try:
                    ids, mask, pos, raw_ids, p_len = self._build_partial_inputs(
                        prompt_messages=prompt_messages,
                        wrong_text=wrong_text,
                        part_response_ratio=sample_part_response_ratio(),
                        max_prompt_length=max_prompt_length,
                        truncation=truncation,
                        pad_token_id=pad_token_id,
                        apply_kwargs=apply_kwargs,
                    )
                except Exception as exc:
                    print(
                        f"[denoise] failed to build partial prompt for problem_id={problem_id!r}: "
                        f"{exc}; falling back to a no-prefix rollout for this slot."
                    )
                    flat_input_ids_rows.append(base_input_ids[b : b + 1])
                    flat_attention_mask_rows.append(base_attention_mask[b : b + 1])
                    flat_position_ids_rows.append(base_position_ids[b : b + 1])
                    flat_raw_prompt_ids.append(
                        list(base_raw_prompt_ids[b]) if base_raw_prompt_ids is not None else []
                    )
                    partial_lens_flat.append(0)
                    n_sub_rows_fallback += 1
                    continue

                flat_input_ids_rows.append(ids)
                flat_attention_mask_rows.append(mask)
                flat_position_ids_rows.append(pos)
                flat_raw_prompt_ids.append(raw_ids)
                partial_lens_flat.append(p_len)
                if p_len > 0:
                    n_sub_rows_with_prefix += 1
                    sum_partial_lens += p_len
                    if p_len > max_observed_partial:
                        max_observed_partial = p_len

        input_ids_t = torch.cat(flat_input_ids_rows, dim=0)
        attention_mask_t = torch.cat(flat_attention_mask_rows, dim=0)
        position_ids_t = torch.cat(flat_position_ids_rows, dim=0)
        raw_prompt_ids_np = np.array(flat_raw_prompt_ids, dtype=object)
        partial_lens_np = np.array(partial_lens_flat, dtype=np.int64)

        gen_batch_td = TensorDict(
            {
                "input_ids": input_ids_t,
                "attention_mask": attention_mask_t,
                "position_ids": position_ids_t,
            },
            batch_size=[B * nk],
        )
        gen_batch_combined = DataProto(
            batch=gen_batch_td,
            non_tensor_batch={"raw_prompt_ids": raw_prompt_ids_np},
            meta_info=dict(gen_batch.meta_info),
        )

        # Single rollout call for all N + K trajectories.
        with marked_timer("gen", timing_raw, "red"):
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_combined)
            timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
            gen_batch_output.meta_info.pop("timing", None)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise NotImplementedError("REMAX is not supported in denoise.")

        # Assign one uid per problem, then repeat to align with N + K rollouts so GRPO groups
        # all (N + K) trajectories of the same problem together.
        new_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
        )
        new_batch = new_batch.repeat(repeat_times=nk, interleave=True)
        new_batch.non_tensor_batch["partial_response_len"] = partial_lens_np
        new_batch = new_batch.union(gen_batch_output)

        # Fold the partial-wrong prefix into the response window. Three modes:
        #   * "shift":   response width = R, partial_wrong gets response_mask = 1
        #                (gradient flows through the off-policy prefix; more signal,
        #                potentially less stable). Trailing p_i rollout tokens are
        #                discarded to enforce partial_wrong + kept <= R.
        #   * "cutdown": response width = R, partial_wrong gets response_mask = 0
        #                (no gradient on the prefix; more stable, less signal).
        #                Trailing p_i rollout tokens are discarded.
        #   * "none":    response width = R + max_partial_len, no truncation
        #                (per-row output length is p_i + R), partial_wrong gets
        #                response_mask = 0. NOT length-fair vs main rollouts but
        #                preserves all generated tokens for comparison.
        max_partial_len = int(partial_lens_np.max()) if partial_lens_np.size > 0 else 0
        if max_partial_len > 0:
            if partial_mode == "none":
                new_batch = self._fold_partial_extended(
                    new_batch, partial_lens_np, max_partial_len
                )
            else:
                # Surface the cost of the R-budget cap: how many real (non-pad)
                # rollout tokens get discarded by the per-row right-shift. Reported
                # for shift/cutdown since they share the same length-truncation.
                _attn = new_batch.batch["attention_mask"]
                _T = _attn.shape[1]
                _plens_t = torch.from_numpy(partial_lens_np).to(
                    device=_attn.device, dtype=torch.long
                )
                _col_pos = torch.arange(_T, device=_attn.device).unsqueeze(0)
                _drop_mask = _col_pos >= (_T - _plens_t.unsqueeze(1))
                _dropped_real_per_row = ((_attn > 0) & _drop_mask).sum(dim=-1)
                dropped_tokens_sum = int(_dropped_real_per_row.sum().item())
                n_rows_with_drops = int((_dropped_real_per_row > 0).sum().item())
                _metrics["denoise/sub_rollout/dropped_tokens_sum"] = float(
                    dropped_tokens_sum
                )
                _metrics["denoise/sub_rollout/n_rows_with_drops"] = float(
                    n_rows_with_drops
                )
                if n_sub_rows_with_prefix > 0:
                    _metrics[
                        "denoise/sub_rollout/dropped_tokens_mean_per_prefix_row"
                    ] = float(dropped_tokens_sum) / float(n_sub_rows_with_prefix)

                new_batch = self._fold_partial_into_response(
                    new_batch,
                    partial_lens_np,
                    max_partial_len,
                    mask_partial_wrong=(partial_mode == "cutdown"),
                )

        _metrics["denoise/sub_rollout/n_main_rows"] = float(B * n)
        _metrics["denoise/sub_rollout/n_sub_rows_planned"] = float(B * k)
        _metrics["denoise/sub_rollout/n_sub_rows_with_prefix"] = float(n_sub_rows_with_prefix)
        _metrics["denoise/sub_rollout/n_sub_rows_fallback"] = float(n_sub_rows_fallback)
        _metrics["denoise/sub_rollout/n_problems_without_wrongs"] = float(problems_without_wrongs)
        _metrics["denoise/sub_rollout/max_partial_len"] = float(max_partial_len)
        if n_sub_rows_with_prefix > 0:
            _metrics["denoise/sub_rollout/mean_partial_len"] = (
                float(sum_partial_lens) / float(n_sub_rows_with_prefix)
            )
            _metrics["denoise/sub_rollout/max_observed_partial_len"] = float(max_observed_partial)

        if self.config.algorithm.use_kl_in_reward:
            raise NotImplementedError("use_kl_in_reward is not supported for denoise.")

        # Optional: tag each row's actor loss group / multiplier so the grouped
        # actor loss treats main vs. sub rollouts as two separate expectations.
        # We discriminate by the *actual* presence of a partial-wrong prefix
        # (``partial_response_len > 0``) rather than slot position: sub-rollout
        # slots that fell back to a no-prefix rollout (because no wrong solution
        # was available, see ``n_sub_rows_fallback``) functionally behave like
        # main rollouts and are joined into the main-rollout loss group here.
        sub_rollout_separate_loss_group = bool(
            self.config.trainer.get("sub_rollout_separate_loss_group", False)
        )
        if k > 0 and sub_rollout_separate_loss_group:
            sub_loss_multiplier = float(
                self.config.trainer.get("sub_rollout_loss_multiplier", 1.0)
            )
            is_sub = partial_lens_np > 0
            new_batch.non_tensor_batch["loss_group_id"] = np.where(
                is_sub, "sub_rollout", "main_rollout"
            ).astype(object)
            new_batch.non_tensor_batch["loss_multiplier"] = np.where(
                is_sub, sub_loss_multiplier, 1.0
            ).astype(np.float32)
            _metrics["denoise/sub_rollout/loss_group_split"] = 1.0
            _metrics["denoise/sub_rollout/loss_multiplier_sub"] = float(
                sub_loss_multiplier
            )
            _metrics["denoise/sub_rollout/n_rows_loss_group_sub"] = float(
                int(is_sub.sum())
            )
            _metrics["denoise/sub_rollout/n_rows_loss_group_main"] = float(
                int((~is_sub).sum())
            )
        else:
            _metrics["denoise/sub_rollout/loss_group_split"] = 0.0

        # Reward computation (rule + optional RM). The reward manager decodes the response slice,
        # which after the boundary shift naturally includes partial-wrong tokens for sub-rollouts.
        reward_extra_infos_dict: dict = {}
        with marked_timer("reward", timing_raw, "yellow"):
            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                new_batch = new_batch.union(reward_tensor)
            if self.global_steps % self.config.trainer.save_freq == 1:
                local_global_step_save_json = os.path.join(
                    self.config.trainer.default_local_dir,
                    f"global_step_{self.global_steps}/rollout.jsonl",
                )
            else:
                local_global_step_save_json = None
            reward_tensor, reward_extra_infos_dict = compute_reward(
                new_batch, self.reward_fn, rollout_save_path=local_global_step_save_json
            )
            new_batch.batch["token_level_scores"] = reward_tensor
            if reward_extra_infos_dict:
                new_batch.non_tensor_batch.update(
                    {key: np.array(v) for key, v in reward_extra_infos_dict.items()}
                )
            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

        return new_batch, reward_extra_infos_dict

    # -------------------------------------------------------------------------
    # Core: advantage + backward
    # -------------------------------------------------------------------------
    def _compute_advantage_and_backward(self, batch, metrics, timing_raw, reward_extra_infos_dict):
        """DP balance / pad, KL-related log probs, advantage, critic/actor updates."""
        def _get_mesh_dp_size(worker_group, mesh_name: str) -> int:
            if mesh_name not in worker_group._dispatch_info:
                worker_group._dispatch_info[mesh_name] = worker_group._query_dispatch_info(mesh_name)
            dp_rank_mapping = worker_group._dispatch_info[mesh_name]
            return max(dp_rank_mapping) + 1

        original_batch_size = batch.batch["attention_mask"].shape[0]
        pad_keep_mask_key = "__dp_pad_keep_mask__"
        size_divisor = _get_mesh_dp_size(self.actor_rollout_wg, "actor")
        if self.use_critic:
            size_divisor = math.lcm(size_divisor, _get_mesh_dp_size(self.critic_wg, "critic"))
        batch, pad_size = pad_dataproto_to_divisor(batch, size_divisor=size_divisor)
        if pad_size > 0:
            padded_batch_size = batch.batch["attention_mask"].shape[0]
            pad_keep_mask = np.zeros(padded_batch_size, dtype=bool)
            pad_keep_mask[:original_batch_size] = True
            batch.non_tensor_batch[pad_keep_mask_key] = pad_keep_mask
            if "attention_mask" in batch.batch:
                batch.batch["attention_mask"][original_batch_size:] = 0
            if "response_mask" in batch.batch:
                batch.batch["response_mask"][original_batch_size:] = 0
            if "token_level_rewards" in batch.batch:
                batch.batch["token_level_rewards"][original_batch_size:] = 0
            if "token_level_scores" in batch.batch:
                batch.batch["token_level_scores"][original_batch_size:] = 0
            if "uid" in batch.non_tensor_batch:
                uid_arr = np.asarray(batch.non_tensor_batch["uid"], dtype=object).copy()
                for pad_idx in range(original_batch_size, padded_batch_size):
                    uid_arr[pad_idx] = f"__dp_pad_uid_{self.global_steps}_{pad_idx - original_batch_size}"
                batch.non_tensor_batch["uid"] = uid_arr
            metrics["train/pad_size_for_dp_divisibility"] = pad_size
            metrics["train/original_batch_size_before_pad"] = original_batch_size
            metrics["train/padded_batch_size_for_dp"] = padded_batch_size

        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        if not self.config.algorithm.use_kl_in_reward:
            batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

        if self.use_critic:
            with marked_timer("values", timing_raw, "cyan"):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch)
            metrics.update(is_metrics)

        with marked_timer("adv", timing_raw, "brown"):
            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            # ``num_repeat`` is the per-problem rollout count used by some advantage estimators.
            # With sub-rollouts, each problem has N + K trajectories under a single uid.
            num_repeat = (
                int(self.config.actor_rollout_ref.rollout.n)
                + int(self.config.trainer.get("sub_rollout_k", 0))
            )
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=num_repeat,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )

        if self.use_critic:
            with marked_timer("update_critic", timing_raw, "pink"):
                critic_output = self.critic_wg.update_critic(batch)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update(critic_output_metrics)

        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, "red"):
                batch.meta_info["actual_global_batch_size"] = original_batch_size
                batch.meta_info["reference_batch_size"] = (
                    self.config.data.train_batch_size
                    * (
                        int(self.config.actor_rollout_ref.rollout.n)
                        + int(self.config.trainer.get("sub_rollout_k", 0))
                    )
                )
                metrics["train/actual_global_batch_size"] = batch.meta_info["actual_global_batch_size"]
                metrics["train/reference_batch_size"] = batch.meta_info["reference_batch_size"]
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)

        if pad_size > 0:
            keep_mask = np.asarray(batch.non_tensor_batch[pad_keep_mask_key], dtype=bool)
            batch = batch.select_idxs(keep_mask)
            batch.non_tensor_batch.pop(pad_keep_mask_key, None)

        return batch, metrics

    # -------------------------------------------------------------------------
    # DAPO: filtering by problem_id avg acc
    # -------------------------------------------------------------------------
    def _dapo_filter_kept_problems(self, batch: DataProto):
        """
        DAPO-style dynamic sampling: aggregate verifier ``acc`` by ``problem_id`` and
        drop **all** rollouts of problems whose average accuracy is exactly ``0`` or
        ``1`` (within ``tol = 1e-5``). Returns the filtered batch, the kept problem
        count, and a metrics dict describing the filtering step.

        Rows of the same ``problem_id`` are kept as a contiguous chunk so a later
        ``[:train_batch_size * (N + K)]`` slice keeps whole problems together.
        """
        acc_vals = batch.non_tensor_batch.get("acc", None)
        problem_ids = batch.non_tensor_batch.get("problem_id", None)
        if acc_vals is None or problem_ids is None:
            raise ValueError(
                "DAPO filtering requires 'acc' and 'problem_id' in non_tensor_batch; "
                f"got acc={acc_vals is not None}, problem_id={problem_ids is not None}."
            )

        acc_arr = np.asarray(acc_vals, dtype=np.float32)
        pid_arr = np.asarray(problem_ids)
        n_rows = len(batch.batch)
        if len(acc_arr) != n_rows or len(pid_arr) != n_rows:
            raise ValueError(
                "DAPO filter shape mismatch: "
                f"acc={len(acc_arr)}, problem_id={len(pid_arr)}, n_rows={n_rows}."
            )

        # Walk rows in order to preserve the contiguous-per-problem layout downstream.
        order_pids: list = []
        pid_to_rows: dict = {}
        pid_to_accs: dict = {}
        for i in range(n_rows):
            pid = pid_arr[i]
            if pid not in pid_to_rows:
                pid_to_rows[pid] = []
                pid_to_accs[pid] = []
                order_pids.append(pid)
            pid_to_rows[pid].append(i)
            pid_to_accs[pid].append(float(acc_arr[i]))

        tol = 1e-5
        kept_idxs: list = []
        n_kept = 0
        for pid in order_pids:
            vals = pid_to_accs[pid]
            avg = sum(vals) / len(vals)
            if not np.isfinite(avg):
                continue
            if abs(avg) <= tol or abs(avg - 1.0) <= tol:
                continue
            kept_idxs.extend(pid_to_rows[pid])
            n_kept += 1

        if kept_idxs:
            filtered = batch[kept_idxs]
        else:
            filtered = batch[np.array([], dtype=np.int64)]

        filter_metrics = {
            "denoise/dapo/n_problems_total": float(len(order_pids)),
            "denoise/dapo/n_problems_kept": float(n_kept),
            "denoise/dapo/n_problems_dropped": float(len(order_pids) - n_kept),
            "denoise/dapo/n_rows_total": float(n_rows),
            "denoise/dapo/n_rows_kept": float(len(kept_idxs)),
        }
        return filtered, n_kept, filter_metrics

    # -------------------------------------------------------------------------
    # uid / loss-group relabeling (shared by both train_batch and DAPO path)
    # -------------------------------------------------------------------------
    def _relabel_uids_and_split_groups(self, batch: DataProto, metrics: dict) -> DataProto:
        """
        Apply optional uid relabeling (``use_problem_id_as_uid``, ``use_same_uid``) and
        optional sub-rollout adv-uid split. Writes related metrics into ``metrics``.

        Must run AFTER rollout/reward and BEFORE ``_compute_advantage_and_backward``
        because the uid identity drives GRPO grouping.
        """
        use_problem_id_as_uid = self.config.trainer.get("use_problem_id_as_uid", False)
        use_same_uid = self.config.trainer.get("use_same_uid", False)

        # GRPO grouping options:
        #   * ``use_problem_id_as_uid``: relabel uid with problem_id (still per-problem
        #     grouping, but human-readable).
        #   * ``use_same_uid``: every row gets same uid (REINFORCE++).
        #   * default: keep the per-problem uuids assigned in ``_rollout_and_compute_reward``
        #     so the N + K trajectories of a problem share a single advantage baseline.
        if use_problem_id_as_uid:
            batch.non_tensor_batch["uid"] = np.array(
                batch.non_tensor_batch["problem_id"], dtype=object
            )
        elif use_same_uid:
            batch.non_tensor_batch["uid"] = np.array(
                ["response_to_problem" for _ in range(len(batch.batch))], dtype=object
            )

        # Optional: split main vs. sub-rollouts into separate advantage groups
        # by suffixing the uid with "_main" / "_sub" (applied AFTER the existing
        # uid relabeling, so it composes with both branches above). The
        # discriminator is the *actual* presence of a partial-wrong prefix
        # (``partial_response_len > 0``): sub-rollout slots that fell back to
        # a no-prefix rollout are treated as main here, matching the
        # ``loss_group_id`` tagging in ``_rollout_and_compute_reward``.
        # ``_balance_batch`` later reorders rows but carries this field along.
        sub_rollout_separate_adv_uid = bool(
            self.config.trainer.get("sub_rollout_separate_adv_uid", False)
        )
        k_sub = int(self.config.trainer.get("sub_rollout_k", 0))
        if k_sub > 0 and sub_rollout_separate_adv_uid:
            plens = batch.non_tensor_batch.get("partial_response_len", None)
            if plens is None:
                raise RuntimeError(
                    "sub_rollout_separate_adv_uid=True requires "
                    "'partial_response_len' in non_tensor_batch (it is set by "
                    "_rollout_and_compute_reward)."
                )
            is_sub_row = np.asarray(plens, dtype=np.int64) > 0
            uids_arr = np.asarray(batch.non_tensor_batch["uid"], dtype=object).copy()
            suffixes = np.where(is_sub_row, "_sub", "_main")
            batch.non_tensor_batch["uid"] = np.array(
                [f"{u}{s}" for u, s in zip(uids_arr, suffixes)], dtype=object
            )
            metrics["denoise/sub_rollout/adv_uid_split"] = 1.0
            metrics["denoise/sub_rollout/n_rows_adv_uid_sub"] = float(
                int(is_sub_row.sum())
            )
            metrics["denoise/sub_rollout/n_rows_adv_uid_main"] = float(
                int((~is_sub_row).sum())
            )
        else:
            metrics["denoise/sub_rollout/adv_uid_split"] = 0.0

        return batch

    # -------------------------------------------------------------------------
    # Core: per-step orchestration
    # -------------------------------------------------------------------------
    def train_batch(
        self,
        batch_dict,
        prev_step_profile,
        curr_step_profile,
        timing_raw,
    ):
        """
        One training step: a single rollout (``N + K`` trajectories per problem), reward
        computation, then PPO advantage + actor/critic update.
        """
        metrics: dict = {}

        with marked_timer("start_profile", timing_raw):
            self._start_profiling(
                not prev_step_profile and curr_step_profile
                if self.config.global_profiler.profile_continuous_steps
                else curr_step_profile
            )

        with marked_timer("step", timing_raw):
            with marked_timer("train_batch/rollout_reward", timing_raw, "red"):
                batch, reward_extra_infos_dict = self._rollout_and_compute_reward(
                    batch_dict, metrics, timing_raw
                )

            batch = self._relabel_uids_and_split_groups(batch, metrics)

            with marked_timer("train_batch/ppo_backward", timing_raw, "pink"):
                batch, metrics = self._compute_advantage_and_backward(
                    batch, metrics, timing_raw, reward_extra_infos_dict
                )

        return batch, metrics

    # -------------------------------------------------------------------------
    # Core: DAPO accumulate-then-update orchestration
    # -------------------------------------------------------------------------
    def _dapo_step(
        self,
        batch_dict,
        prev_step_profile,
        curr_step_profile,
        timing_raw,
        dapo_state: dict,
    ):
        """
        One iteration of the DAPO dynamic-sampling loop.

        Per call:
          1. Run a single rollout + reward on the freshly-yielded ``gen_batch`` (size
             ``data.gen_batch_size``).
          2. Filter the resulting rows by aggregating verifier ``acc`` per
             ``problem_id`` and dropping problems whose average acc is exactly
             ``0`` or ``1``.
          3. Concatenate the kept rows into ``dapo_state["batch"]`` and update the
             ``num_kept_problems`` / ``num_gen_batches`` counters.

        If the accumulated kept-problem count is still ``< train_batch_size``, return
        ``None`` to signal the caller to consume another gen_batch.

        Once ``>= train_batch_size``, slice the first ``train_batch_size * (N + K)``
        rows (whole-problem chunks), apply uid relabeling, run advantage + PPO
        backward, and return ``(batch, metrics)``. The DAPO state is reset before
        returning.

        Profiling is started on the FIRST gen_batch of each global step (when the
        accumulator is empty) so that a single ``_start_profiling`` / ``_stop_profiling``
        pair brackets the entire accumulate-and-update span.
        """
        starting_new_step = dapo_state.get("batch") is None
        if starting_new_step:
            with marked_timer("start_profile", timing_raw):
                self._start_profiling(
                    not prev_step_profile and curr_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )

        # Reuse the accumulated metrics dict across gen_batches so we keep DAPO
        # counters AND the latest rollout metrics for the caller to log.
        metrics: dict = dapo_state.setdefault("metrics", {})

        with marked_timer("step", timing_raw):
            with marked_timer("train_batch/rollout_reward", timing_raw, "red"):
                new_batch, reward_extra_infos_dict = self._rollout_and_compute_reward(
                    batch_dict, metrics, timing_raw
                )
            dapo_state["reward_extra_infos_dict"] = reward_extra_infos_dict
            dapo_state["num_gen_batches"] = int(dapo_state.get("num_gen_batches", 0)) + 1

            filtered_batch, n_kept, filter_metrics = self._dapo_filter_kept_problems(new_batch)
            # DAPO counters are summed across gen_batches within a single global step
            # so the user can see the cumulative cost of getting to train_batch_size.
            for k, v in filter_metrics.items():
                metrics[k] = float(metrics.get(k, 0.0)) + float(v)

            if dapo_state.get("batch") is None:
                dapo_state["batch"] = filtered_batch
            elif len(filtered_batch.batch) > 0:
                dapo_state["batch"] = DataProto.concat(
                    [dapo_state["batch"], filtered_batch]
                )
            dapo_state["num_kept_problems"] = (
                int(dapo_state.get("num_kept_problems", 0)) + int(n_kept)
            )

            prompt_bsz = int(self.config.data.train_batch_size)
            num_kept = int(dapo_state["num_kept_problems"])
            num_gen = int(dapo_state["num_gen_batches"])

            metrics["denoise/dapo/num_gen_batches"] = float(num_gen)
            metrics["denoise/dapo/num_kept_problems_cumulative"] = float(num_kept)

            if num_kept < prompt_bsz:
                # Not enough problems yet; cap the number of attempts to avoid getting
                # stuck on a degenerate dataset (matches baseline DAPO behavior).
                max_num_gen_batches = int(
                    self.config.trainer.get("dapo_max_num_gen_batches", 0)
                )
                if 0 < max_num_gen_batches <= num_gen:
                    raise ValueError(
                        f"DAPO: num_gen_batches={num_gen} >= "
                        f"dapo_max_num_gen_batches={max_num_gen_batches} but only "
                        f"num_kept_problems={num_kept} < train_batch_size={prompt_bsz}. "
                        "Increase data.gen_batch_size, raise dapo_max_num_gen_batches, "
                        "or set it to 0 to disable the cap."
                    )
                print(
                    f"[denoise DAPO] num_kept_problems={num_kept} < "
                    f"train_batch_size={prompt_bsz}; rolling another gen_batch "
                    f"(num_gen_batches={num_gen})."
                )
                return None

            # We have enough kept problems. Slice to exactly train_batch_size whole
            # problems. Each kept problem contributes (N + K) contiguous rollout rows
            # in self._dapo_filter_kept_problems, so a flat row slice keeps groups
            # intact.
            nk = (
                int(self.config.actor_rollout_ref.rollout.n)
                + int(self.config.trainer.get("sub_rollout_k", 0))
            )
            traj_bsz = prompt_bsz * nk
            batch = dapo_state["batch"]
            if len(batch.batch) < traj_bsz:
                raise RuntimeError(
                    f"DAPO accounting error: num_kept_problems={num_kept} >= "
                    f"train_batch_size={prompt_bsz} but accumulated rows "
                    f"{len(batch.batch)} < {traj_bsz} = train_batch_size * (N + K). "
                    "Did a kept problem contribute fewer than (N + K) rows?"
                )
            batch = batch[:traj_bsz]

            metrics["denoise/dapo/n_rows_used_in_update"] = float(traj_bsz)
            metrics["denoise/dapo/n_problems_used_in_update"] = float(prompt_bsz)
            reward_extra_infos_dict = dapo_state.get("reward_extra_infos_dict", {})

            # Reset accumulation BEFORE the heavy backward step so even an exception
            # there leaves the state clean for the next call.
            dapo_state["batch"] = None
            dapo_state["num_kept_problems"] = 0
            dapo_state["num_gen_batches"] = 0
            dapo_state["metrics"] = {}
            dapo_state["reward_extra_infos_dict"] = {}

            batch = self._relabel_uids_and_split_groups(batch, metrics)

            with marked_timer("train_batch/ppo_backward", timing_raw, "pink"):
                batch, metrics = self._compute_advantage_and_backward(
                    batch, metrics, timing_raw, reward_extra_infos_dict
                )

        return batch, metrics
