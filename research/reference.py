from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from .data import ROOT

REFERENCE = ROOT / "runs" / "a8630092fa9e"


def artifact_bytes(directory: Path, name: str) -> bytes:
    path = directory / name
    if name == "predictions.json" and not path.exists():
        return gzip.decompress(path.with_suffix(".json.gz").read_bytes())
    return path.read_bytes()


def load_reference(directory: Path = REFERENCE) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256(artifact_bytes(directory, name)).hexdigest() != expected:
            raise ValueError(f"Reference checksum mismatch: {name}")
    return {
        "manifest": manifest,
        **{name: json.loads(artifact_bytes(directory, f"{name}.json"))
           for name in ("protocol", "frozen", "summary", "splits")},
        "rows": json.loads(artifact_bytes(directory, "predictions.json")),
    }
