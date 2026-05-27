# in_memory_dataset.py
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from .rl_dataset import RLHFDataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing import Optional, List, Dict, Any
import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
import copy
import logging
import os
import re
import traceback
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def _normalize_row_for_arrow(row: dict) -> dict:
    """
    Force string type for problem/answer and nested text fields so PyArrow
    does not infer int64 (or other wrong types) when building the Dataset.
    """
    row = dict(row)
    # Top-level text fields
    for key in ("problem", "answer", "question","action","super_uid","data_source","ability"):
        if key in row and row[key] is not None:
            row[key] = str(row[key])
        elif key in row:
            row[key] = ""
    for key in ("level","problem_id"):
        if key in row and row[key] is not None:
            row[key] = int(row[key])
        elif key in row:
            row[key] = 0
    # extra_info: question, answer (and any other text that might be mixed type)
    if "extra_info" in row and isinstance(row["extra_info"], dict):
        ei = dict(row["extra_info"])
        for key in ("question", "answer", "split", "level"):
            if key in ei and ei[key] is not None:
                ei[key] = str(ei[key])
            elif key in ei:
                ei[key] = ""
        row["extra_info"] = ei
    # reward_model.ground_truth
    if "reward_model" in row and isinstance(row["reward_model"], dict):
        rm = dict(row["reward_model"])
        if "ground_truth" in rm and rm["ground_truth"] is not None:
            rm["ground_truth"] = str(rm["ground_truth"])
        elif "ground_truth" in rm:
            rm["ground_truth"] = ""
        row["reward_model"] = rm
    # prompt: ensure each message content is str when it's scalar (avoid list/multimodal)
    if "prompt" in row and isinstance(row["prompt"], list):
        def _ensure_content_str(msg):
            if not isinstance(msg, dict) or "content" not in msg:
                return msg
            c = msg["content"]
            if c is None:
                return {**msg, "content": ""}
            if isinstance(c, (list, dict)):
                return msg  # leave multimodal content as-is
            return {**msg, "content": str(c)}
        row["prompt"] = [_ensure_content_str(msg) for msg in row["prompt"]]
    return row


class InMemoryRLHFDataset(RLHFDataset):
    def __init__(
        self,
        data_list: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):  
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get("image_patch_size", 14)
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        # Mirror RLHFDataset: append a string to the last user message at __getitem__ time
        # via the inherited `_build_messages`. Empty disables the rewrite.
        self.user_prompt_suffix = config.get("user_prompt_suffix", "") or ""

        self.tool_config_path = config.get("tool_config_path", None)
        self.tool_schemas = None
        if self.tool_config_path:
            try:
                from verl.tools.utils.tool_registry import initialize_tools_from_config

                tool_list = initialize_tools_from_config(self.tool_config_path)
                # match ToolAgentLoop behaviour: model_dump to plain dicts
                self.tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
            except Exception as e:
                logger.warning("Failed to initialize tools from %s: %s", self.tool_config_path, e)
                self.tool_schemas = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self._has_logged_continue_final_message = False
        
        
        self.get_dataframe()

        
    def get_dataframe(self):
        # Normalize so problem/answer and nested text fields are always str;
        # avoids PyArrow inferring int64 and failing when later rows have string values.
        normalized_list = [_normalize_row_for_arrow(row) for row in self.data_list]
        self.dataframe = HFDataset.from_list(normalized_list)
        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(
                    self._apply_chat_template_safe(
                        tokenizer,
                        doc[prompt_key],
                        add_generation_prompt=True,
                        tokenize=True,
                        apply_kwargs=self._get_apply_kwargs_for_doc(doc),
                    )
                )
                <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def _get_apply_kwargs_for_doc(self, doc: dict) -> dict:
        apply_kwargs = dict(**self.apply_chat_template_kwargs)
        per_sample_kwargs = doc.get("_apply_chat_template_kwargs", None)
        if isinstance(per_sample_kwargs, dict):
            apply_kwargs.update(per_sample_kwargs)
        return apply_kwargs

    @staticmethod
    def _apply_chat_template_safe(processing_class, messages, add_generation_prompt: bool, tokenize: bool, apply_kwargs: dict):
        continue_final_message = bool(apply_kwargs.get("continue_final_message", False))
        clean_kwargs = {k: v for k, v in apply_kwargs.items() if k != "continue_final_message"}

        def _safe_apply(msgs, add_gen_prompt, tok, **kw):
            try:
                return processing_class.apply_chat_template(
                    msgs, add_generation_prompt=add_gen_prompt, tokenize=tok, **kw
                )
            except TypeError:
                return processing_class.apply_chat_template(
                    msgs, add_generation_prompt=add_gen_prompt, tokenize=tok
                )

        if continue_final_message and messages and messages[-1].get("role") == "assistant":
            # Manually build the prompt for partial-response continuation.
            # Do NOT pass continue_final_message to the template, because some
            # templates (e.g. Qwen3) insert <think></think> around the content,
            # breaking the continuation semantics. Instead:
            #   1. Render all messages except the final assistant with add_generation_prompt=True
            #      → produces "...<|im_start|>assistant\n"
            #   2. Directly append the partial response text
            partial_content = messages[-1].get("content", "")
            prefix_messages = list(messages[:-1])
            rendered_prefix = _safe_apply(prefix_messages, True, False, **clean_kwargs)
            full_text = rendered_prefix + partial_content
            print(f"rendered_prefix:{rendered_prefix}")
            print(f"partial_content:{partial_content}")
            if tokenize:
                _tokenizer = getattr(processing_class, "tokenizer", processing_class)
                return _tokenizer.encode(full_text, add_special_tokens=False)
            return full_text

        return _safe_apply(messages, add_generation_prompt, tokenize, **clean_kwargs)

    def _maybe_log_continue_mode(self, apply_kwargs: dict, raw_prompt: str):
        if self._has_logged_continue_final_message:
            return
        if not apply_kwargs.get("continue_final_message", False):
            return
        print(
            "[chat_template continue] continue_final_message=True, "
            f"add_generation_prompt=True, prompt_tail={(raw_prompt)}"
        )
        self._has_logged_continue_final_message = True

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            apply_kwargs = self._get_apply_kwargs_for_doc(row_dict)
            raw_prompt = self._apply_chat_template_safe(
                self.processor, messages, add_generation_prompt=True, tokenize=False, apply_kwargs=apply_kwargs
            )
            self._maybe_log_continue_mode(apply_kwargs, raw_prompt)
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [process_image(image, image_patch_size=self.image_patch_size) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            videos_kwargs = {}
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos, video_metadata = zip(
                    *[
                        process_video(video, image_patch_size=self.image_patch_size, return_video_metadata=True)
                        for video in row_dict_videos
                    ],
                    strict=True,
                )
                videos = list(videos)
                video_metadata = list(video_metadata)
                videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [
                    (video.numpy(), metadata) for video, metadata in zip(videos, video_metadata, strict=True)
                ]

            model_inputs = self.processor(
                text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs, return_tensors="pt"
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            apply_kwargs = self._get_apply_kwargs_for_doc(row_dict)
            raw_prompt = self._apply_chat_template_safe(
                self.tokenizer, messages, add_generation_prompt=True, tokenize=False, apply_kwargs=apply_kwargs
            )
            self._maybe_log_continue_mode(apply_kwargs, raw_prompt)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict