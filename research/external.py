from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
from typing import Iterable
from urllib.request import urlopen

import numpy as np
import pyarrow.parquet as pq

from .data import FEATURES, ROOT, REVISION, digest, features

FORMAT = "calibri-multipl-e-python-external-v1"
CONFIG = "multipl-e_qwen3"
NAMESPACE = "multipl-e:py:"
EXPECTED_TASKS = {"train": 80, "validation": 40, "test": 41}
SOURCES = {
    "train": {"bytes": 23860599, "sha256": "b7673b3cf0932ee88c3ff5144e50f999b3e93199b8676098ab950a570ae24a4d"},
    "validation": {"bytes": 9404450, "sha256": "81def9d7ac0739c3041858d1d28cb5e2b52700cb9a193577e4f4cf8c2e4504f2"},
    "test": {"bytes": 12907039, "sha256": "c0e13313a190fe4f6e42fc4c027103b23578be4b0c84a6d076f44f3174f6b4a0"},
}
for split, source in SOURCES.items():
    source["filename"] = f"{split}-00000-of-00001.parquet"
    source["url"] = f"https://huggingface.co/datasets/lavis-nlp/CALIBRI/resolve/{REVISION}/{CONFIG}/{source['filename']}"

UPSTREAM_NOTICE = {
    "filename": "MultiPL-E-LICENSE.txt",
    "url": "https://raw.githubusercontent.com/nuprl/MultiPL-E/3025a531af7450e7df8b96fe0440e9804480bbad/LICENSE",
    "bytes": 1846,
    "sha256": "6891bf32c7df59f0665d45a22bc09224ed122394eae4404048e0e4c7e4bc2f23",
}


def fetch_verified(source: dict, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / source["filename"]
    if path.exists():
        if path.stat().st_size != source["bytes"] or digest(path) != source["sha256"]:
            raise ValueError(f"Cached source does not match its pin: {path.name}")
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    with urlopen(source["url"], timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if temporary.stat().st_size != source["bytes"] or digest(temporary) != source["sha256"]:
        raise ValueError(f"Downloaded source does not match its pin: {path.name}")
    temporary.replace(path)
    return path


def flatten_python_rows(records: Iterable[dict], source_split: str, seen: set[str]) -> list[dict]:
    if source_split not in EXPECTED_TASKS:
        raise ValueError("Unknown official source split")
    rows = []
    for record in records:
        if record.get("language") != "py":
            continue
        task = record.get("id")
        prompt = record.get("prompt")
        if not isinstance(task, str) or not task or not isinstance(prompt, str) or not prompt:
            raise ValueError("Python tasks require nonempty IDs and prompts")
        task_id = NAMESPACE + task
        if task_id in seen:
            raise ValueError(f"Duplicate Python task across source rows or splits: {task}")
        if record.get("model") != "qwen3-coder":
            raise ValueError(f"Unexpected primary model for {task}")
        fields = (record.get("program"), record.get("is_correct"), record.get("token_logprobs"))
        if any(not isinstance(value, list) or len(value) != 10 for value in fields):
            raise ValueError(f"Expected ten aligned candidates for {task}")
        if any(not isinstance(code, str) for code in record["program"]):
            raise ValueError(f"Programs must be strings for {task}")
        if any(type(label) is not bool for label in record["is_correct"]):
            raise ValueError(f"Correctness labels must be booleans for {task}")
        if any(not isinstance(tokens, list) or any(not isinstance(pair, list) or len(pair) != 2
                                                 or any(not isinstance(value, str) for value in pair) for pair in tokens)
               for tokens in record["token_logprobs"]):
            raise ValueError(f"Malformed token-score pairs for {task}")
        fingerprint = hashlib.sha256(prompt.encode()).hexdigest()
        normalized = hashlib.sha256(" ".join(prompt.split()).encode()).hexdigest()
        candidates = [{"task_id": task_id, "candidate": index,
                       "features": features(prompt, record["program"][index], record["token_logprobs"][index]),
                       "label": int(record["is_correct"][index]), "prompt_sha256": fingerprint,
                       "normalized_prompt_sha256": normalized, "source_split": source_split}
                      for index in range(10)]
        seen.add(task_id)
        rows.extend(candidates)
    return rows


def prepare_external(output: Path, protocol_path: Path, cache: Path | None = None) -> dict:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("Use a separate output directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    protocol_bytes = protocol_path.read_bytes()
    if not isinstance(json.loads(protocol_bytes), dict):
        raise ValueError("The frozen protocol must be a JSON object")
    source_hashes = {name: digest(ROOT / name) for name in ("data.py", "external.py")}
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_bytes(protocol_bytes)
    cache = ROOT / "data" / CONFIG if cache is None else cache
    source_records, rows, seen = [], [], set()
    prompt_splits = {}
    for split, source in SOURCES.items():
        path = fetch_verified(source, cache)
        split_rows = []
        columns = ["id", "prompt", "language", "model", "program", "is_correct", "token_logprobs"]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=columns):
            split_rows.extend(flatten_python_rows(batch.to_pylist(), split, seen))
        tasks = [row for row in split_rows if row["candidate"] == 0]
        if len(tasks) != EXPECTED_TASKS[split]:
            raise ValueError(f"Python task count changed in {split}")
        for task in tasks:
            previous = prompt_splits.setdefault(task["normalized_prompt_sha256"], split)
            if previous != split:
                raise ValueError("Duplicate normalized Python prompt across source splits")
        source_records.append({"split": split, "config": CONFIG, **source, "python_tasks": len(tasks)})
        rows.extend(split_rows)
    notice = fetch_verified(UPSTREAM_NOTICE, cache)
    (output / "MultiPL-E-LICENSE.txt").write_bytes(notice.read_bytes())
    if source_hashes != {name: digest(ROOT / name) for name in source_hashes}:
        raise ValueError("Extraction source changed during the run; restart and use a new output path")
    payload = {"feature_names": FEATURES, "rows": rows}
    (output / "rows.json.gz").write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
    manifest = {
        "format": FORMAT, "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "evaluation_only", "dataset": "lavis-nlp/CALIBRI", "revision": REVISION,
        "config": CONFIG, "language": "py", "model": "qwen3-coder", "namespace": NAMESPACE,
        "tasks": len(seen), "rows": len(rows), "candidates_per_task": 10,
        "tasks_by_source_split": dict(Counter(row["source_split"] for row in rows if row["candidate"] == 0)),
        "feature_names": FEATURES, "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "source_files": source_records, "extractor_sha256": source_hashes,
        "prompt_fingerprints": {"prompt_sha256": "SHA-256 of original prompt UTF-8 bytes",
                                "normalized_prompt_sha256": "SHA-256 of UTF-8 single-space-joined prompt.split()"},
        "environment": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "pyarrow")}},
        "notices": {
            "calibri_card": f"https://huggingface.co/datasets/lavis-nlp/CALIBRI/blob/{REVISION}/README.md",
            "calibri_license_declaration": "MIT, as declared by the dataset card; upstream benchmark conditions remain relevant",
            "upstream_license": UPSTREAM_NOTICE,
            "upstream_terms": "BSD 3-Clause with Machine Learning Restriction: retain copyright, conditions and disclaimer; no use as ML training data",
            "use_restriction": "No fitting, score calibration, method selection or label-dependent filtering on these external tasks",
            "publication": "Keep this per-row extraction local; publish aggregate evaluation results unless redistribution rights have been reviewed separately",
        },
        "files": {name: digest(output / name) for name in ("protocol.json", "rows.json.gz", "MultiPL-E-LICENSE.txt")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_external(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("format") != FORMAT or manifest.get("scope") != "evaluation_only"
            or manifest.get("revision") != REVISION or manifest.get("config") != CONFIG):
        raise ValueError("Unsupported external extraction")
    for name in ("protocol.json", "rows.json.gz", "MultiPL-E-LICENSE.txt"):
        if digest(directory / name) != manifest["files"][name]:
            raise ValueError(f"External extraction hash mismatch: {name}")
    if digest(directory / "protocol.json") != manifest["protocol_sha256"]:
        raise ValueError("External extraction protocol hash mismatch")
    payload = json.loads(gzip.decompress((directory / "rows.json.gz").read_bytes()))
    rows = payload["rows"]
    if payload["feature_names"] != FEATURES or len(rows) != manifest["rows"]:
        raise ValueError("External extraction schema or row count changed")
    identities = [(row["task_id"], row["candidate"]) for row in rows]
    tasks = Counter(row["task_id"] for row in rows)
    if len(set(identities)) != len(rows) or len(tasks) != manifest["tasks"] or set(tasks.values()) != {10}:
        raise ValueError("External extraction task identities changed")
    if any(not row["task_id"].startswith(NAMESPACE) or row["candidate"] not in range(10)
           or type(row["label"]) is not int or row["label"] not in (0, 1)
           or len(row["features"]) != len(FEATURES) or not np.isfinite(row["features"]).all()
           for row in rows):
        raise ValueError("External extraction contains invalid numerical rows")
    return {"rows": rows, "manifest": manifest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pinned MultiPL-E Python tasks for frozen evaluation only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    result = prepare_external(args.output, args.protocol, args.cache)
    print(json.dumps({"tasks": result["tasks"], "rows": result["rows"], "protocol_sha256": result["protocol_sha256"], "scope": result["scope"]}, indent=2))
