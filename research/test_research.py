from __future__ import annotations

import json
import unittest

import numpy as np
from scipy.special import expit, logit

from .data import FEATURES, ROOT, digest, features
from .methods import (audit, bin_ids, budget_threshold, clustered_interval, export_classifier,
                      fit_classifier, fit_patches, membership, policy, predict_classifier, replay, route_mask)


class ResearchChecks(unittest.TestCase):
    def test_features_never_execute_code(self):
        result = features("sample", 'raise RuntimeError("must not execute")', [["-0.1", "raise"]])
        self.assertEqual(len(result), 16)
        self.assertTrue(np.isfinite(result).all())

    def test_invalid_and_missing_features(self):
        values = features("sample", "def (", [])
        self.assertEqual(values[FEATURES.index("syntax_valid")], 0)
        self.assertEqual(values[FEATURES.index("missing_logprobs")], 1)
        self.assertEqual(features("sample", "", [])[FEATURES.index("empty_code")], 1)
        with self.assertRaises(ValueError):
            features("sample", "x=1", [["nan", "x"]])

    def test_current_bin_and_clipping(self):
        fit = {"bins": 5, "patches": [{"group": "all", "bin": 4, "delta": -.3},
                                      {"group": "all", "bin": 2, "delta": -.8}]}
        actual = replay(np.array([.82, 1.]), np.ones((2, 1), bool), ["all"], fit)
        np.testing.assert_allclose(actual, [0, .7])
        np.testing.assert_array_equal(bin_ids(np.array([0., .2, 1.]), 5), [0, 1, 4])

    def test_task_support_not_candidate_support(self):
        scores = np.repeat(.8, 100)
        labels = np.zeros(100)
        groups = np.ones((100, 1), bool)
        ids = np.repeat("same_task", 100)
        cells = audit(scores, labels, groups, ids, ["all"], 5, 20)
        self.assertFalse(cells[0]["supported"])
        fit = fit_patches(scores, labels, groups, ids, ["all"], 5, 20, .03, 10)
        self.assertEqual(fit["status"], "no_supported_cells")
        self.assertEqual(fit["patches"], [])

    def test_fit_removes_known_residual(self):
        p, y = np.repeat(.8, 40), np.tile([0., 1.], 20)
        groups = np.ones((40, 1), bool)
        fit = fit_patches(p, y, groups, np.arange(40), ["all"], 5, 20, .01, 20)
        result = replay(p, groups, ["all"], fit)
        np.testing.assert_allclose(result, .5)
        self.assertEqual(fit["status"], "converged_on_fit_cells")

    def test_tied_scores_respect_budget_without_labels(self):
        scores, keys = np.zeros(100), np.arange(100) / 100
        cutoff = budget_threshold(scores, .3, keys)
        self.assertEqual(route_mask(scores, cutoff, keys).sum(), 30)
        self.assertEqual(route_mask(scores, budget_threshold(scores, 0, keys), keys).sum(), 0)
        self.assertEqual(route_mask(scores, budget_threshold(scores, 1, keys), keys).sum(), 100)

    def test_no_accepted_answers_has_undefined_error(self):
        result = policy(np.zeros(3), np.array([0, 1, 0]), np.array([1, 0, 0]),
                        {"score": 1.000001, "tie_key": 0}, np.array([.1, .2, .3]))
        self.assertIsNone(result["accepted_error"])
        self.assertAlmostEqual(result["success"], 1/3)
        self.assertAlmostEqual(result["rescued"], 1/3)
        self.assertAlmostEqual(result["regressed"], 1/3)

    def test_cluster_interval_does_not_gain_precision_from_copies(self):
        x, ids = np.arange(20) / 20, np.arange(20)
        one = clustered_interval(x, ids, 73129, 300)
        repeated = clustered_interval(np.repeat(x, 10), np.repeat(ids, 10), 73129, 300)
        np.testing.assert_allclose([one["low"], one["high"]], [repeated["low"], repeated["high"]])

    def test_model_roundtrip(self):
        rng = np.random.default_rng(123)
        x = rng.normal(size=(200, 3))
        y = (x[:, 0] + rng.normal(size=200) > 0).astype(int)
        model = fit_classifier(x, y)
        artifact = json.loads(json.dumps(export_classifier(model, ["a", "b", "c"])))
        np.testing.assert_allclose(predict_classifier(artifact, x), model.predict_proba(x)[:, 1], atol=1e-12)


class RecordedExperimentChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pointer = ROOT / "runs" / "latest.json"
        if not pointer.exists():
            raise unittest.SkipTest("Run python -m research.run for recorded-data integration checks")
        cls.directory = ROOT / "runs" / json.loads(pointer.read_text())["run_id"]
        cls.summary = json.loads((cls.directory / "summary.json").read_text())
        cls.frozen = json.loads((cls.directory / "frozen.json").read_text())
        cls.rows = json.loads((cls.directory / "predictions.json").read_text())

    def test_frozen_artifacts_and_split_integrity(self):
        manifest = json.loads((self.directory / "manifest.json").read_text())
        for name, expected in manifest["files"].items():
            self.assertEqual(digest(self.directory / name), expected)
        splits = json.loads((self.directory / "splits.json").read_text())
        ids = [task for tasks in splits.values() for task in tasks]
        self.assertEqual(len(ids), 1055)
        self.assertEqual(len(set(ids)), 1055)
        for split, task_ids in splits.items():
            self.assertEqual(sum(row["split"] == split for row in self.rows), len(task_ids)*10)

    def test_label_free_replay_matches_every_saved_score(self):
        x = np.array([row["features"] for row in self.rows])
        frozen = self.frozen
        names = frozen["group_names"]
        groups = membership(x, frozen["group_thresholds"], names)
        models = frozen["models"]
        logistic = predict_classifier(models["logistic"], x)
        platt = expit(models["platt"]["coefficient"] * logit(np.clip(logistic, 1e-6, 1-1e-6)) + models["platt"]["intercept"])
        corrected = replay(platt, groups, names, models["multicalibration"])
        for method, actual in (("logistic", logistic), ("platt", platt), ("multicalibration", corrected)):
            np.testing.assert_allclose(actual, [row["scores"][method] for row in self.rows], atol=1e-12)

    def test_validation_not_test_selects_corrections(self):
        for method in ("multicalibration", "score_bins"):
            candidates = [row for row in self.summary["selection"] if row["method"] == method]
            best = min(candidates, key=lambda row: (row["validation_brier"], row["updates"]))
            self.assertEqual(best["epsilon"], self.frozen["models"][method]["epsilon"])
            self.assertEqual(best["support"], self.frozen["models"][method]["support"])

    def test_validation_budget_for_all_baselines(self):
        rows = [row for row in self.rows if row["split"] == "validation"]
        keys = np.array([row["tie_key"] for row in rows])
        for method, cutoff in self.frozen["routing_thresholds"].items():
            scores = np.array([row["scores"][method] for row in rows])
            self.assertEqual(int(route_mask(scores, cutoff, keys).sum()), int(.3 * len(rows)))

    def test_reported_brier_and_paired_effect(self):
        rows = [row for row in self.rows if row["split"] == "test"]
        y = np.array([row["label"] for row in rows])
        for method, result in self.summary["partitions"]["test"]["methods"].items():
            p = np.array([row["scores"][method] for row in rows])
            self.assertTrue(np.isfinite(p).all())
            self.assertAlmostEqual(float(np.mean((p-y)**2)), result["brier"])
        metrics = self.summary["partitions"]["test"]["methods"]
        self.assertAlmostEqual(metrics["multicalibration"]["brier"] - metrics["platt"]["brier"],
                               self.summary["comparisons"]["multicalibration"]["estimate"])


if __name__ == "__main__":
    unittest.main()
