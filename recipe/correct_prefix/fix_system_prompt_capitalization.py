#!/usr/bin/env python3
"""Fix the capitalization of the default system prompt in Parquet datasets.

By default, every ``*.parquet`` file under ``data/`` is scanned recursively.
Only system messages whose content exactly equals the old prompt are changed;
all other values are left untouched. Files are rewritten atomically and the
operation is idempotent, so running the script more than once is safe.

Examples
--------
    # Preview the changes in the repository's data directory.
    python recipe/correct_prefix/fix_system_prompt_capitalization.py --dry-run

    # Update the repository's data directory in place.
    python recipe/correct_prefix/fix_system_prompt_capitalization.py

    # Update one file or a different data directory.
    python recipe/correct_prefix/fix_system_prompt_capitalization.py /path/to/data
"""

from __future__ import annotations

import argparse
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


OLD_SYSTEM_PROMPT = "you are a helpful assistant."
NEW_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True)
class FileResult:
    path: Path
    rows: int
    affected_rows: int
    replacements: int
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将 Parquet 的 prompt 列中精确匹配的 system 文本 "
            f"{OLD_SYSTEM_PROMPT!r} 替换为 {NEW_SYSTEM_PROMPT!r}。"
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("data")],
        help="要处理的 Parquet 文件或目录；目录会递归扫描（默认: data）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计将发生的替换，不写入文件",
    )
    return parser.parse_args()


def find_parquet_files(paths: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"路径不存在: {path}")
        if path.is_file():
            if path.suffix.lower() != ".parquet":
                raise ValueError(f"不是 Parquet 文件: {path}")
            files.add(path.resolve())
        elif path.is_dir():
            files.update(candidate.resolve() for candidate in path.rglob("*.parquet"))
        else:
            raise ValueError(f"既不是文件也不是目录: {path}")

    if not files:
        raise FileNotFoundError("指定路径下没有找到 .parquet 文件")
    return sorted(files)


def rewrite_prompts(prompts: Sequence[Any], source: Path) -> tuple[list[Any], int, int]:
    rewritten: list[Any] = []
    affected_rows = 0
    replacements = 0

    for row_index, prompt in enumerate(prompts):
        if prompt is None:
            rewritten.append(None)
            continue
        if not isinstance(prompt, list):
            raise ValueError(
                f"{source}: prompt 第 {row_index} 行应为消息列表，实际为 "
                f"{type(prompt).__name__}"
            )

        new_prompt: list[dict[str, Any]] = []
        row_replacements = 0
        for message_index, message in enumerate(prompt):
            if not isinstance(message, dict):
                raise ValueError(
                    f"{source}: prompt 第 {row_index} 行第 {message_index} 条消息应为 dict，"
                    f"实际为 {type(message).__name__}"
                )

            new_message = dict(message)
            if (
                new_message.get("role") == "system"
                and new_message.get("content") == OLD_SYSTEM_PROMPT
            ):
                new_message["content"] = NEW_SYSTEM_PROMPT
                row_replacements += 1
            new_prompt.append(new_message)

        if row_replacements:
            affected_rows += 1
            replacements += row_replacements
        rewritten.append(new_prompt)

    return rewritten, affected_rows, replacements


def parquet_compression(path: Path) -> str | None:
    """Return the existing uniform codec so rewrites retain file compression."""
    parquet_file = pq.ParquetFile(path)
    codecs = {
        parquet_file.metadata.row_group(row_group).column(column).compression
        for row_group in range(parquet_file.metadata.num_row_groups)
        for column in range(parquet_file.metadata.row_group(row_group).num_columns)
    }
    if len(codecs) != 1:
        # Mixed codecs cannot be represented by a single write_table argument.
        # Snappy is PyArrow's default and matches the datasets in this repository.
        return "snappy"
    codec = codecs.pop().lower()
    return None if codec == "uncompressed" else codec


def write_atomically(table: pa.Table, destination: Path, compression: str | None) -> None:
    original_mode = stat.S_IMODE(destination.stat().st_mode)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        pq.write_table(table, temporary_path, compression=compression)

        # Verify the complete temporary file before replacing the original.
        verified = pq.read_table(temporary_path)
        if verified.num_rows != table.num_rows or not verified.schema.equals(
            table.schema, check_metadata=True
        ):
            raise RuntimeError(f"写入校验失败，未覆盖原文件: {destination}")

        os.chmod(temporary_path, original_mode)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def process_file(path: Path, dry_run: bool) -> FileResult:
    table = pq.read_table(path)
    if "prompt" not in table.column_names:
        return FileResult(path, table.num_rows, 0, 0, "skipped: no prompt column")

    prompt_index = table.schema.get_field_index("prompt")
    prompt_field = table.schema.field(prompt_index)
    prompt_type = prompt_field.type
    rewritten, affected_rows, replacements = rewrite_prompts(
        table.column(prompt_index).to_pylist(), path
    )

    if replacements == 0:
        return FileResult(path, table.num_rows, 0, 0, "unchanged")
    if dry_run:
        return FileResult(path, table.num_rows, affected_rows, replacements, "dry-run")

    new_prompt_column = pa.array(rewritten, type=prompt_type)
    updated_table = table.set_column(prompt_index, prompt_field, new_prompt_column)
    if not updated_table.schema.equals(table.schema, check_metadata=True):
        raise RuntimeError(f"替换导致 Schema 发生变化，未写入文件: {path}")
    write_atomically(updated_table, path, parquet_compression(path))
    return FileResult(path, table.num_rows, affected_rows, replacements, "updated")


def main() -> None:
    args = parse_args()
    files = find_parquet_files(args.paths)
    results = [process_file(path, args.dry_run) for path in files]

    for result in results:
        print(
            f"[{result.status.upper()}] {result.path}: rows={result.rows}, "
            f"affected_rows={result.affected_rows}, replacements={result.replacements}"
        )

    print(
        "\nSummary: "
        f"files={len(results)}, rows={sum(result.rows for result in results)}, "
        f"affected_rows={sum(result.affected_rows for result in results)}, "
        f"replacements={sum(result.replacements for result in results)}, "
        f"mode={'dry-run' if args.dry_run else 'in-place'}"
    )


if __name__ == "__main__":
    main()
