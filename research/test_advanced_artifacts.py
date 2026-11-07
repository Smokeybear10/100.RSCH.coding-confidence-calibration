from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from .data import ROOT, digest
from .reference import load_reference

PROJECT = ROOT.parent


class AdvancedArtifactChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = PROJECT / "advanced_results"
        cls.simulations = PROJECT / "simulation_results"
        cls.summary = json.loads((cls.results / "summary.json").read_text())
        cls.protocol = json.loads((cls.results / "protocol.json").read_text())

    def test_numerical_manifests_and_core_sources_match_full_runs(self):
        for directory in (self.results, self.simulations):
            manifest = json.loads((directory / "manifest.json").read_text())
            for name, expected in manifest["files"].items():
                self.assertEqual(digest(directory / name), expected)
        self.assertFalse(self.summary["smoke"])
        self.assertEqual(self.summary["folds"], len(self.protocol["seeds"])*self.protocol["outer_folds"])
        self.assertEqual(digest(self.results / "protocol.json"), self.summary["protocol_sha256"])
        for name in ("advanced.py", "auditors.py", "simulations.py"):
            self.assertEqual(digest(ROOT / name), self.summary["code_sha256"][f"research/{name}"])
        simulation = json.loads((self.simulations / "simulation.json").read_text())
        self.assertFalse(simulation["smoke"])
        self.assertEqual(digest(self.simulations / "protocol.json"), simulation["protocol_sha256"])

    def test_out_of_fold_rows_match_source_labels_and_task_counts(self):
        frame = pd.read_csv(self.results / "predictions.csv.gz")
        reference = load_reference()
        source = pd.DataFrame([{"task": row["task_id"], "candidate": row["candidate"],
                                "expected_y": row["label"], "expected_alternate": row["alternate_label"]}
                               for row in reference["rows"]])
        joined = frame.merge(source, on=["task", "candidate"], validate="many_to_one")
        self.assertEqual(len(joined), self.summary["prediction_rows"])
        np.testing.assert_array_equal(joined["y"], joined["expected_y"])
        np.testing.assert_array_equal(joined["alternate"], joined["expected_alternate"])
        np.testing.assert_array_equal(frame.groupby(["seed", "method", "task"]).size(), 10)
        self.assertTrue(np.isfinite(frame["p"]).all())
        self.assertTrue(frame["p"].between(0, 1).all())
        self.assertTrue(frame["estimated_benefit"].between(-1, 1).all())
        for _, current in frame.groupby(["seed", "method"]):
            self.assertEqual(current["task"].nunique(), self.summary["task_clusters"])
        expected = len(reference["rows"])*len(self.protocol["seeds"])*len(frame["method"].unique())
        self.assertEqual(len(frame), expected)

    def test_task_cost_archive_recomputes_reported_objectives(self):
        task_costs = pd.read_csv(self.results / "task_costs.csv.gz")
        costs = pd.read_csv(self.results / "decision_costs.csv")
        keys = ["seed", "fold", "method", "selector", "failure_loss"]
        measured = task_costs.groupby(keys, as_index=False)["objective"].mean()
        joined = measured.merge(costs[keys+["total_objective"]], on=keys, validate="one_to_one")
        self.assertEqual(len(joined), len(costs))
        self.assertEqual(len(task_costs), self.summary["task_cost_rows"])
        np.testing.assert_allclose(joined["objective"], joined["total_objective"], rtol=0, atol=1e-12)
        selected = task_costs[task_costs["failure_loss"] == 20]
        benefit = selected[(selected["method"] == "benefit_model") & (selected["selector"] == "predicted_benefit")].groupby("task")["objective"].mean()
        baseline = selected[(selected["method"] == "logistic/platt") & (selected["selector"] == "all_validation_boundaries")].groupby("task")["objective"].mean()
        contrasts = pd.read_csv(self.results / "cost_comparisons.csv")
        row = contrasts[(contrasts["method"] == "benefit_model") & (contrasts["selector"] == "predicted_benefit") & (contrasts["failure_loss"] == 20)].iloc[0]
        self.assertAlmostEqual(float((benefit-baseline).mean()), row["objective_difference"], places=12)

    def test_simulation_summary_matches_individual_seed_records(self):
        frame = pd.read_csv(self.simulations / "simulation_runs.csv")
        summary = pd.read_csv(self.simulations / "simulation_summary.csv")
        keys = ["scenario", "calibration_tasks", "method"]
        actual = frame.groupby(keys, as_index=False)["known_probability_mse"].mean()
        paired = actual.merge(summary[keys+["known_probability_mse"]], on=keys, suffixes=("", "_saved"), validate="one_to_one")
        np.testing.assert_allclose(paired["known_probability_mse"], paired["known_probability_mse_saved"], atol=1e-12)
        self.assertEqual(frame["seed"].nunique(), len(self.protocol["simulation"]["seeds"]))
        self.assertTrue((frame.query("scenario == 'calibrated_null' and method == 'raw'")["known_probability_mse"] == 0).all())


if __name__ == "__main__":
    unittest.main()
