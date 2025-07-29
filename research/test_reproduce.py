from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from .methods import route_mask
from .reference import REFERENCE, load_reference
from .reproduce import partitions_from_reference, refit_models, replay_models, reproduce, result_tables


class ReproductionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = load_reference()
        cls.parts = partitions_from_reference(cls.reference)
        cls.scores = replay_models(cls.parts, cls.reference["frozen"])

    def test_every_saved_probability_replays_without_outcomes(self):
        changed = {name: {**part, "y": 1-part["y"], "alternate": 1-part["alternate"]}
                   for name, part in self.parts.items()}
        independent = replay_models(changed, self.reference["frozen"])
        for split, methods in self.scores.items():
            for method, scores in methods.items():
                expected = [row["scores"][method] for row in self.parts[split]["rows"]]
                np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(scores, independent[split][method])

    def test_recorded_memberships_and_candidate_counts(self):
        names = self.reference["frozen"]["group_names"]
        for split, part in self.parts.items():
            np.testing.assert_array_equal(np.unique(part["ids"]), sorted(self.reference["splits"][split]))
            for row, group in zip(part["rows"], part["groups"]):
                self.assertEqual(row["groups"], [name for name, active in zip(names, group) if active])

    def test_recomputed_metrics_and_intervals_match_reference(self):
        tables = result_tables(self.scores, self.parts, self.reference["frozen"], self.reference["protocol"])
        for row in tables["metrics"]:
            expected = self.reference["summary"]["partitions"][row["split"]]["methods"][row["method"]]
            for metric in ("brier", "log_loss"):
                self.assertAlmostEqual(row[metric], expected[metric], places=12)
        for row in tables["comparisons"]:
            expected = self.reference["summary"]["comparisons"][row["method"]]
            np.testing.assert_allclose([row[key] for key in ("estimate", "low", "high")],
                                       [expected[key] for key in ("estimate", "low", "high")], atol=1e-12)

    def test_original_score_bins_reproduce_original_diagnostics(self):
        original = {split: {method: np.array([row["scores"][method] for row in part["rows"]])
                            for method in self.scores[split]}
                    for split, part in self.parts.items()}
        tables = result_tables(original, self.parts, self.reference["frozen"], self.reference["protocol"])
        for row in tables["metrics"]:
            expected = self.reference["summary"]["partitions"][row["split"]]["methods"][row["method"]]
            for metric in ("ece", "worst"):
                self.assertAlmostEqual(row[metric], expected[metric], places=12)

    def test_refit_does_not_use_test_outcomes(self):
        scores, frozen = refit_models(self.parts, self.reference["protocol"])
        changed = {name: {**part} for name, part in self.parts.items()}
        changed["test"]["y"] = 1-changed["test"]["y"]
        changed["test"]["alternate"] = 1-changed["test"]["alternate"]
        independent, other = refit_models(changed, self.reference["protocol"])
        self.assertEqual(frozen, other)
        for method, p in scores["test"].items():
            np.testing.assert_array_equal(p, independent["test"][method])
            expected = [row["scores"][method] for row in self.parts["test"]["rows"]]
            np.testing.assert_allclose(p, expected, rtol=0, atol=1e-8)

    def test_platt_preserves_recorded_routing_decisions(self):
        test = self.parts["test"]
        thresholds = self.reference["frozen"]["routing_thresholds"]
        raw = route_mask(self.scores["test"]["logistic"], thresholds["logistic"], test["keys"])
        platt = route_mask(self.scores["test"]["platt"], thresholds["platt"], test["keys"])
        np.testing.assert_array_equal(raw, platt)

    def test_reference_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "reference"
            shutil.copytree(REFERENCE, directory, ignore=shutil.ignore_patterns("predictions.json"))
            with (directory / "frozen.json").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch: frozen.json"):
                load_reference(directory)

    def test_outputs_are_separate_and_cannot_overwrite_previous_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "replay"
            report = reproduce(directory)
            self.assertTrue(report["all_probabilities_match"])
            self.assertEqual(report["table_rows"]["metrics"], 36)
            self.assertEqual(json.loads((directory / "protocol.json").read_text()), self.reference["protocol"])
            with self.assertRaises(FileExistsError):
                reproduce(directory)
        with self.assertRaisesRegex(ValueError, "outside the saved reference"):
            reproduce(REFERENCE)


if __name__ == "__main__":
    unittest.main()
