"""
简单的 validation 脚本（基于 vLLM 离线推理）。

流程
----
1. 读入一个 parquet 数据集（``prompt`` 列是 chat messages，形如
   ``[{"role": "system", "content": ...}, {"role": "user", "content": ...}]``；
   ``reward_model`` 列里有 ``ground_truth``）。
2. 对每条样本用模型采样 ``rollout_n`` 次。
3. 从每条 rollout 文本里提取最后一个 ``\\boxed{...}``：
       - 若提取得到内容 ``X``，则用 ``solution_str = "\\boxed{X}"`` 调用
         :func:`verl.utils.reward_score.think_test_math.compute_score` 打分；
       - 若整段输出没有任何 ``\\boxed{}``，则该条 rollout 直接判 0 分（视为答错）。
4. 保存 rollout 明细到 ``rollouts.jsonl`` / ``rollouts.parquet``。
5. 额外把原 parquet 复制一份，并新增一列 ``wrong_answer_with_boxed``
   （``list[str]``，存放该样本所有「含 boxed 但答错」的 rollout 全文；若没有
   则为空列表），落到 ``<output-dir>/<原文件名>.with_wrong_boxed.parquet``。
6. 控制台 + ``summary.json`` 报告：
       - 模型的总体正确率（micro，所有 rollout 上的平均）；
       - 每个样本至少有一个 rollout 答错的概率；
       - 至少有一个「含 boxed 但答错」rollout 的样本数。

用法示例
--------
    python recipe/denoise/data_prepare.py \\
        --model /path/to/hf-model \\
        --dataset /path/to/val.parquet \\
        --rollout-n 16 \\
        --output-dir ./val_outputs/exp1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from recipe.denoise.verifier import extract_last_boxed  # noqa: E402
from verl.utils.reward_score.think_test_math import compute_score  # noqa: E402


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _ensure_message_list(prompt: Any) -> List[dict]:
    """将 parquet 中的 ``prompt`` 字段统一成 ``list[dict]``（兼容 numpy.ndarray / tuple）。"""
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, tuple):
        prompt = list(prompt)
    if not isinstance(prompt, list):
        raise ValueError(
            f"prompt 必须是消息列表 list[dict]，实际类型: {type(prompt).__name__}"
        )
    out: List[dict] = []
    for m in prompt:
        if isinstance(m, np.ndarray):
            m = m.tolist()
        if not isinstance(m, dict):
            raise ValueError(f"消息项必须是 dict，实际类型: {type(m).__name__}")
        out.append({"role": str(m["role"]), "content": m["content"]})
    return out


def _get_ground_truth(row: pd.Series) -> str:
    """从 ``reward_model`` 列中取 ``ground_truth`` 字符串。"""
    rm = row.get("reward_model")
    if hasattr(rm, "tolist") and not isinstance(rm, dict):
        rm = rm.tolist()
    if isinstance(rm, dict):
        gt = rm.get("ground_truth")
        if gt is None:
            raise ValueError("reward_model 缺少 ground_truth 字段")
        return str(gt)
    raise ValueError(
        f"reward_model 必须是 dict，实际类型: {type(rm).__name__}"
    )


def _to_float_acc(acc_val: Any) -> float:
    """``compute_score`` 返回的 acc 可能是 bool / numpy.bool_ / int / float，统一成 0/1 float。"""
    try:
        return 1.0 if float(acc_val) > 0 else 0.0
    except Exception:
        return 1.0 if bool(acc_val) else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="对模型在 parquet 数据集上做 rollout 验证，并按 boxed 答案打分。",
    )
    p.add_argument("--model", required=True, help="模型路径（HF 目录或 HF 名）")
    p.add_argument(
        "--dataset", required=True, type=Path, help="parquet 数据集路径（含 prompt / reward_model）"
    )
    p.add_argument("--rollout-n", type=int, default=8, help="每条样本采样次数（默认 8）")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./val_outputs"),
        help="结果保存目录（默认 ./val_outputs）",
    )

    # 采样参数
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)

    # vLLM 引擎参数
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")

    # 调试 / 范围
    p.add_argument(
        "--limit", type=int, default=None, help="仅取前 N 条样本（调试用，默认全部）"
    )
    p.add_argument(
        "--save-parquet",
        action="store_true",
        help="额外把 rollouts 落成 parquet（rollouts.parquet）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # vLLM / transformers 都很重，延迟 import
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # -------- 1. 加载数据 --------
    df = pd.read_parquet(args.dataset)
    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)
    n_samples = len(df)
    if n_samples == 0:
        raise RuntimeError(f"数据集为空: {args.dataset}")

    print(
        f"[INFO] 数据集: {args.dataset}\n"
        f"[INFO] 样本数: {n_samples}    rollout_n: {args.rollout_n}\n"
        f"[INFO] 模型:    {args.model}"
    )

    # -------- 2. 应用 chat 模板 --------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    prompts_text: List[str] = []
    ground_truths: List[str] = []
    for i in range(n_samples):
        row = df.iloc[i]
        msgs = _ensure_message_list(row["prompt"])
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts_text.append(text)
        ground_truths.append(_get_ground_truth(row))

    # -------- 3. 启动 vLLM --------
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        n=args.rollout_n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # -------- 4. 推理 --------
    outputs = llm.generate(prompts_text, sampling_params)

    # -------- 5. 打分 + 汇总 --------
    rollout_records: List[dict] = []          # 给 jsonl 用（保留每条 rollout 全文）
    flat_rows: List[dict] = []                # 给 parquet 用（每行一条 rollout）
    # 每个样本对应一个 list[str]，存所有「含 boxed 但答错」的 rollout 全文。
    # 长度恒等于 n_samples，与原 df 一一对应；没有就给空列表。
    per_sample_wrong_with_boxed: List[List[str]] = []
    per_sample_mean_acc: List[float] = []     # 每个样本的平均 acc
    per_sample_any_wrong: List[float] = []    # 每个样本是否至少有一个 rollout 答错
    per_sample_boxed_rate: List[float] = []   # 每个样本里成功提取到 boxed 的比例
    total_rollouts = 0
    total_correct = 0

    for i, output in enumerate(tqdm(outputs, desc="Scoring")):
        gt = ground_truths[i]
        sample_outputs = output.outputs  # List[CompletionOutput] (len == rollout_n)

        rollouts: List[dict] = []
        accs: List[float] = []
        boxed_hits = 0
        wrong_with_boxed_texts: List[str] = []

        for j, gen in enumerate(sample_outputs):
            response_text = gen.text
            boxed = extract_last_boxed(response_text)

            if boxed is None:
                solution_str = ""
                score_dict = {
                    "score": 0.0,
                    "acc": 0,
                    "answer": "",
                    "pred": "",
                    "format_verify": 0.0,
                }
            else:
                boxed_hits += 1
                solution_str = "\\boxed{" + boxed + "}"
                score_dict = compute_score(solution_str, gt)

            acc = _to_float_acc(score_dict.get("acc", 0))
            score = float(score_dict.get("score", 0.0))
            n_resp_tokens = (
                len(gen.token_ids) if getattr(gen, "token_ids", None) is not None else None
            )

            rollouts.append(
                {
                    "rollout_idx": j,
                    "response": response_text,
                    "boxed": boxed,
                    "solution_str": solution_str,
                    "score": score,
                    "acc": acc,
                    "finish_reason": getattr(gen, "finish_reason", None),
                    "num_response_tokens": n_resp_tokens,
                }
            )
            accs.append(acc)
            flat_rows.append(
                {
                    "sample_idx": i,
                    "rollout_idx": j,
                    "ground_truth": gt,
                    "response": response_text,
                    "boxed": boxed,
                    "solution_str": solution_str,
                    "acc": acc,
                    "score": score,
                }
            )

            # 收集「含 boxed 但答错」的 rollout 全文
            if boxed is not None and acc < 1.0:
                wrong_with_boxed_texts.append(response_text)

        accs_arr = np.array(accs, dtype=np.float32)
        mean_acc = float(accs_arr.mean())
        any_wrong = float((accs_arr < 1.0).any())
        boxed_rate = boxed_hits / len(sample_outputs) if sample_outputs else 0.0

        per_sample_mean_acc.append(mean_acc)
        per_sample_any_wrong.append(any_wrong)
        per_sample_boxed_rate.append(boxed_rate)
        per_sample_wrong_with_boxed.append(wrong_with_boxed_texts)
        total_rollouts += len(sample_outputs)
        total_correct += int(accs_arr.sum())

        rollout_records.append(
            {
                "sample_idx": i,
                "prompt_text": prompts_text[i],
                "ground_truth": gt,
                "mean_acc": mean_acc,
                "any_wrong": any_wrong,
                "boxed_extract_rate": boxed_rate,
                "rollouts": rollouts,
            }
        )

    overall_acc = total_correct / total_rollouts if total_rollouts else 0.0
    prob_any_wrong = (
        float(np.mean(per_sample_any_wrong)) if per_sample_any_wrong else 0.0
    )
    mean_boxed_rate = (
        float(np.mean(per_sample_boxed_rate)) if per_sample_boxed_rate else 0.0
    )

    # -------- 6. 落盘 --------
    rollout_jsonl = args.output_dir / "rollouts.jsonl"
    with rollout_jsonl.open("w", encoding="utf-8") as f:
        for rec in rollout_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 在原 parquet 上新增 `wrong_answer_with_boxed` 列后另存
    df_out = df.copy()
    df_out["wrong_answer_with_boxed"] = pd.Series(
        per_sample_wrong_with_boxed, index=df_out.index, dtype=object
    )
    augmented_parquet = args.output_dir / (args.dataset.stem + ".with_wrong_boxed.parquet")
    df_out.to_parquet(augmented_parquet, index=False)

    if args.save_parquet:
        rollout_parquet = args.output_dir / "rollouts.parquet"
        pd.DataFrame(flat_rows).to_parquet(rollout_parquet, index=False)
    else:
        rollout_parquet = None

    n_with_wrong_boxed = int(sum(1 for xs in per_sample_wrong_with_boxed if xs))
    total_wrong_boxed = int(sum(len(xs) for xs in per_sample_wrong_with_boxed))

    summary = {
        "model": args.model,
        "dataset": str(args.dataset),
        "n_samples": n_samples,
        "rollout_n": args.rollout_n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "overall_acc": overall_acc,
        "prob_at_least_one_wrong": prob_any_wrong,
        "mean_boxed_extract_rate": mean_boxed_rate,
        "num_samples_with_wrong_boxed": n_with_wrong_boxed,
        "total_wrong_boxed_rollouts": total_wrong_boxed,
        "per_sample_mean_acc": per_sample_mean_acc,
        "per_sample_any_wrong": per_sample_any_wrong,
        "per_sample_boxed_extract_rate": per_sample_boxed_rate,
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # -------- 7. 终端报告 --------
    print("\n=================== 验证结果 ===================")
    print(f"样本数                  : {n_samples}")
    print(f"rollout_n               : {args.rollout_n}")
    print(f"总体正确率(micro)       : {overall_acc:.4f}   ({total_correct}/{total_rollouts})")
    print(f"至少一次错误概率        : {prob_any_wrong:.4f}")
    print(f"boxed 提取成功率        : {mean_boxed_rate:.4f}")
    print(
        f"有 boxed 但答错的样本    : {n_with_wrong_boxed} / {n_samples}"
        f"   ({n_with_wrong_boxed / n_samples:.4f})"
    )
    print(f"有 boxed 但答错的 rollout: {total_wrong_boxed}")
    print(f"明细文件                : {rollout_jsonl}")
    print(f"带新列的 parquet        : {augmented_parquet}")
    if rollout_parquet is not None:
        print(f"明细 parquet            : {rollout_parquet}")
    print(f"汇总文件                : {summary_path}")
    print("===============================================\n")


if __name__ == "__main__":
    main()
