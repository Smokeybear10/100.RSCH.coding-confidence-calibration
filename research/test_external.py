from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from . import external
from .data import FEATURES


def record(task: str = "HumanEval_fixture", language: str = "py") -> dict:
    return {"id": task, "language": language, "model": "qwen3-coder", "prompt": "def fixture(x):\n    return x\n",
            "program": ["raise RuntimeError('must never execute generated code')"] * 10,
            "is_correct": [True, False] * 5, "token_logprobs": [[["-0.2", "return"]]] * 10}


class ExternalExtractionChecks(unittest.TestCase):
    def test_python_filter_and_features_do_not_execute_programs(self):
        rows = external.flatten_python_rows([record(language="java"), record()], "train", set())
        self.assertEqual(len(rows), 10)
        self.assertEqual({row["task_id"] for row in rows}, {external.NAMESPACE + "HumanEval_fixture"})
        self.assertEqual([row["label"] for row in rows], [1, 0] * 5)
        self.assertTrue(all(len(row["features"]) == len(FEATURES) for row in rows))
        prompt = record()["prompt"]
        self.assertEqual(rows[0]["prompt_sha256"], hashlib.sha256(prompt.encode()).hexdigest())
        self.assertEqual(rows[0]["normalized_prompt_sha256"], hashlib.sha256(" ".join(prompt.split()).encode()).hexdigest())
        self.assertNotIn("program", rows[0])
        self.assertNotIn("prompt", rows[0])
        self.assertNotIn("token_logprobs", rows[0])

    def test_duplicate_tasks_across_splits_are_rejected(self):
        seen = set()
        external.flatten_python_rows([record()], "train", seen)
        with self.assertRaisesRegex(ValueError, "Duplicate Python task"):
            external.flatten_python_rows([record()], "test", seen)

    def test_invalid_candidates_and_models_fail_at_the_source_boundary(self):
        variants = []
        for field in ("program", "is_correct", "token_logprobs"):
            item = record()
            item[field] = item[field][:-1]
            variants.append(item)
        integer_labels = record()
        integer_labels["is_correct"] = [1] * 10
        variants.append(integer_labels)
        wrong_model = record()
        wrong_model["model"] = "other-model"
        variants.append(wrong_model)
        invalid_scores = record()
        invalid_scores["token_logprobs"] = [[["nan", "return"]]] * 10
        variants.append(invalid_scores)
        for item in variants:
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    external.flatten_python_rows([item], "train", set())

    def test_nonempty_output_is_preserved_without_network_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "keep.txt").write_text("keep")
            with patch.object(external, "fetch_verified", side_effect=AssertionError("No network")):
                with self.assertRaises(FileExistsError):
                    external.prepare_external(path, path / "missing-protocol.json")
            self.assertEqual((path / "keep.txt").read_text(), "keep")

    def test_protocol_snapshot_notices_and_numeric_cache_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text('{"external_use":"evaluation_only"}\n')
            protocol_bytes = protocol.read_bytes()
            fixtures = {}
            expected = {}
            for index, split in enumerate(external.SOURCES):
                item = record(f"HumanEval_fixture_{index}")
                item["prompt"] = f"def fixture_{index}(x):\n    return x\n"
                path = root / f"{split}.parquet"
                pq.write_table(pa.Table.from_pylist([item]), path)
                fixtures[external.SOURCES[split]["filename"]] = path
                expected[split] = 1
            notice = root / "LICENSE.txt"
            notice.write_text("Synthetic test notice; no external material used")
            fixtures[external.UPSTREAM_NOTICE["filename"]] = notice

            def fetch(source: dict, cache: Path) -> Path:
                self.assertEqual((root / "output" / "protocol.json").read_bytes(), protocol_bytes)
                return fixtures[source["filename"]]

            with patch.object(external, "fetch_verified", side_effect=fetch), patch.object(external, "EXPECTED_TASKS", expected):
                manifest = external.prepare_external(root / "output", protocol)
            loaded = external.load_external(root / "output")
            self.assertEqual(manifest["tasks"], 3)
            self.assertEqual(manifest["rows"], 30)
            self.assertEqual(manifest["scope"], "evaluation_only")
            self.assertEqual(manifest["protocol_sha256"], hashlib.sha256(protocol_bytes).hexdigest())
            self.assertEqual(len(loaded["rows"]), 30)
            self.assertEqual((root / "output" / "MultiPL-E-LICENSE.txt").read_bytes(), notice.read_bytes())
            rows_path = root / "output" / "rows.json.gz"
            rows_path.write_bytes(rows_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                external.load_external(root / "output")

    def test_duplicate_normalized_prompts_across_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text("{}")
            fixtures = {}
            for index, split in enumerate(external.SOURCES):
                item = record(f"HumanEval_fixture_{index}")
                item["prompt"] = "def fixture(x):\n  return x\n" if index else "def fixture(x): return x"
                path = root / f"{split}.parquet"
                pq.write_table(pa.Table.from_pylist([item]), path)
                fixtures[external.SOURCES[split]["filename"]] = path
            with patch.object(external, "fetch_verified", side_effect=lambda source, cache: fixtures[source["filename"]]), patch.object(external, "EXPECTED_TASKS", {split: 1 for split in external.SOURCES}):
                with self.assertRaisesRegex(ValueError, "Duplicate normalized Python prompt"):
                    external.prepare_external(root / "output", protocol)


if __name__ == "__main__":
    unittest.main()
