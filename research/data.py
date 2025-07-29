from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
REVISION = "7a4a7dc7aceb1b4a426c48761d0f128f2daf6a54"
HASHES = {
    "livecodebench_qwen3/train": "3aee997a10af1eafa7dc1e2ee56bea3dfe97fa5cd4cb875146ac36f5fc6f8b07",
    "livecodebench_qwen3/validation": "324c10df81f8d49c7dcd23ace33a8b94b20d963a593308b8230de491e3f8ea4a",
    "livecodebench_qwen3/test": "787bd1bf73a343ce3bee3c89d472b61f4e2312294b3d9aa659742b5ad19920ff",
    "livecodebench_gpt-oss/train": "1d2104cce8d696a7849f5473b7c6a4b7f4bbfbf09a99952bf195ba5b0aa91333",
    "livecodebench_gpt-oss/validation": "9ffae47c020ae14c0b49d30126af876505b71ad320b671fd1dfb1bb1987c8824",
    "livecodebench_gpt-oss/test": "f085d8c6d26ce2e930a1186ef5d52352ed3d98d7ab76fe38a59a89328e9fedfa",
}
FEATURES = [
    "log_prompt_chars", "log_code_chars", "log_code_lines", "log_output_tokens",
    "mean_logprob", "p10_logprob", "logprob_sd", "low_probability_fraction",
    "ast_depth", "branches", "loops", "functions", "imports", "syntax_valid",
    "empty_code", "missing_logprobs",
]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(config: str, split: str) -> Path:
    target = ROOT / "data" / config / f"{split}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = HASHES[f"{config}/{split}"]
    if target.exists() and digest(target) == expected:
        return target
    temporary = target.with_suffix(".part")
    url = f"https://huggingface.co/datasets/lavis-nlp/CALIBRI/resolve/{REVISION}/{config}/{split}-00000-of-00001.parquet"
    print(f"Downloading {config}/{split}", flush=True)
    subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3",
                    "--max-time", "600", "--output", str(temporary), url], check=True)
    if digest(temporary) != expected:
        raise ValueError(f"Source checksum mismatch: {target.name}")
    temporary.replace(target)
    return target


def features(prompt: str, code: str, token_pairs: list) -> list[float]:
    # Parse the program as data only. Generated code is never executed.
    nodes, depth, valid = [], 0, 0
    if code.strip():
        try:
            tree = ast.parse(code)
            valid = 1
            stack = [(tree, 0)]
            while stack:
                node, level = stack.pop()
                nodes.append(node)
                depth = max(depth, level)
                stack.extend((child, level + 1) for child in ast.iter_child_nodes(node))
        except (SyntaxError, ValueError, RecursionError):
            pass
    probabilities = np.array([float(pair[0]) for pair in token_pairs], dtype=float)
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities > 1e-6):
        raise ValueError("Invalid sampled-token log probabilities")
    missing = len(probabilities) == 0
    values = probabilities if not missing else np.array([0.0])
    return [
        math.log1p(len(prompt)), math.log1p(len(code)), math.log1p(len(code.splitlines())),
        math.log1p(len(probabilities)), float(values.mean()), float(np.quantile(values, .1)),
        float(values.std()), float(np.mean(values < -2)), float(depth),
        float(sum(isinstance(n, (ast.If, ast.IfExp, ast.Match)) for n in nodes)),
        float(sum(isinstance(n, (ast.For, ast.While, ast.AsyncFor, ast.comprehension)) for n in nodes)),
        float(sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in nodes)),
        float(sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in nodes)),
        float(valid), float(not code.strip()), float(missing),
    ]


def prepare(protocol: dict) -> dict:
    if (protocol["revision"] != REVISION or protocol["primary"] != "livecodebench_qwen3"
            or protocol["alternate"] != "livecodebench_gpt-oss"):
        raise ValueError("This adapter supports only the pinned LiveCodeBench model pair")
    cache = ROOT / "data" / "prepared.json"
    cache_hash = cache.with_suffix(".sha256")
    source_hash = digest(Path(__file__))
    if cache.exists():
        existing = json.loads(cache.read_text())
        if existing["extractor_hash"] == source_hash and existing["revision"] == REVISION:
            if not cache_hash.exists() or digest(cache) != cache_hash.read_text().strip():
                raise ValueError("The prepared dataset has changed; remove the cache and re-extract from the verified sources")
            if any(digest(ROOT / item["path"]) != item["sha256"] for item in existing["sources"]):
                raise ValueError("A cached source file has changed")
            return existing
    result = {"revision": REVISION, "extractor_hash": source_hash, "sources": [], "tasks": []}
    # Preserve all source rows, including empty and syntactically invalid programs.
    for split in ("train", "validation", "test"):
        primary_path = download(protocol["primary"], split)
        alternate_path = download(protocol["alternate"], split)
        for model, path in ((protocol["primary"], primary_path), (protocol["alternate"], alternate_path)):
            result["sources"].append({"config": model, "split": split, "sha256": digest(path),
                                      "bytes": path.stat().st_size, "path": str(path.relative_to(ROOT))})
        alternate = {}
        for batch in pq.ParquetFile(alternate_path).iter_batches(batch_size=16, columns=["id", "prompt", "program", "is_correct", "model"]):
            for row in batch.to_pylist():
                if row["id"] in alternate:
                    raise ValueError("Duplicate alternate task ID")
                alternate[row["id"]] = row
        seen = set()
        for batch in pq.ParquetFile(primary_path).iter_batches(batch_size=8):
            for row in batch.to_pylist():
                task_id = row["id"]
                if task_id in seen:
                    raise ValueError("Duplicate primary task ID")
                seen.add(task_id)
                other = alternate[task_id]
                if row["prompt"] != other["prompt"]:
                    raise ValueError(f"Model prompt mismatch for {task_id}")
                fields = (row["program"], row["is_correct"], row["token_logprobs"], other["program"], other["is_correct"])
                if any(len(field) != 10 for field in fields):
                    raise ValueError(f"Expected ten candidates for {task_id}")
                if any(type(label) is not bool for labels in (row["is_correct"], other["is_correct"]) for label in labels):
                    raise ValueError("Labels must be booleans")
                task = {"id": task_id, "name": row["name"], "prompt": row["prompt"],
                        "source_split": split, "difficulty": row["difficulty"], "candidates": []}
                for j in range(10):
                    task["candidates"].append({"index": j, "code": row["program"][j],
                        "alternate_code": other["program"][j], "label": int(row["is_correct"][j]),
                        "alternate_label": int(other["is_correct"][j]),
                        "features": features(row["prompt"], row["program"][j], row["token_logprobs"][j])})
                result["tasks"].append(task)
        if seen != set(alternate):
            raise ValueError("Model task sets differ")
        print(f"Extracted {split}: {len(seen)} paired tasks", flush=True)
    ids = [task["id"] for task in result["tasks"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Task leakage across official partitions")
    prompt_splits = {}
    for task in result["tasks"]:
        key = " ".join(task["prompt"].split())
        previous = prompt_splits.setdefault(key, task["source_split"])
        if previous != task["source_split"]:
            raise ValueError("Duplicate prompt across official partitions")
    cache.write_text(json.dumps(result, separators=(",", ":")))
    cache_hash.write_text(digest(cache) + "\n")
    return result
