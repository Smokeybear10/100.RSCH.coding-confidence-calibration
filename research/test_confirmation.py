from __future__ import annotations

import copy
import inspect
import json
import unittest
from unittest.mock import patch

import numpy as np

from .confirmation import METHODS, apply_confirmation, fit_confirmation


FACTORIAL = ("discovery_fixed", "independent_fixed", "discovery_gated", "independent_gated")


def sample(role: str, tasks: int, seed: int) -> dict:
    x = np.random.default_rng(seed).normal(size=(tasks * 2, 16))
    x[:, 1] = np.repeat(np.where(np.arange(tasks) % 2, 1.0, -1.0), 2)
    return {"x": x, "p": np.full(tasks * 2, .5), "y": (x[:, 1] > 0).astype(int),
            "ids": np.repeat([f"{role}:{i}" for i in range(tasks)], 2)}


class ConfirmationChecks(unittest.TestCase):
    def setUp(self):
        self.parts = [sample(role, tasks, seed) for role, tasks, seed in
                      (("D", 80, 1), ("E", 40, 2), ("C", 40, 3), ("V", 40, 4))]
        self.settings = {"bins": 5, "max_depth": 2, "min_samples_leaf": 40,
                         "min_discovery_tasks": 20, "epsilon": .03,
                         "fixed_step": .5, "min_independent_tasks": 8,
                         "bootstrap_replicates": 1000, "alpha": .05,
                         "step_grid": [0, .125, .25, .5, 1]}

    def fit(self, parts: list[dict] | None = None) -> dict:
        return fit_confirmation(*(self.parts if parts is None else parts), self.settings, 17)

    def test_factorial_shares_frozen_rule_step_and_support(self):
        fitted = self.fit()
        self.assertEqual(set(fitted["methods"]), set(METHODS))
        self.assertTrue(fitted["eligible"])
        self.assertTrue(fitted["gate"]["accepted"])
        self.assertEqual(fitted["support"], {"discovery": 40, "estimation": 20, "confirmation": 20})
        matching = fitted["matched_estimation"]
        self.assertEqual(matching["tasks"], 20)
        self.assertEqual(len(matching["discovery_ids"]), 20)
        self.assertEqual(len(matching["estimation_ids"]), 20)
        self.assertEqual(matching["discovery_candidates"], 40)
        self.assertEqual(matching["estimation_candidates"], 40)
        for name in FACTORIAL:
            self.assertEqual(fitted["methods"][name]["rule"], fitted["rule"])
            self.assertEqual(fitted["methods"][name]["step"], .5)
        json.dumps(fitted)

    def test_confirmation_labels_change_gate_but_not_ungated_estimates(self):
        initial = self.fit()
        changed = copy.deepcopy(self.parts)
        changed[2]["y"] = 1 - changed[2]["y"]
        other = self.fit(changed)
        self.assertEqual(initial["rule"], other["rule"])
        self.assertTrue(initial["gate"]["accepted"])
        self.assertFalse(other["gate"]["accepted"])
        for name in ("discovery_fixed", "independent_fixed"):
            self.assertEqual(initial["methods"][name]["delta"], other["methods"][name]["delta"])
            self.assertEqual(initial["methods"][name]["estimate"], other["methods"][name]["estimate"])

    def test_estimation_labels_cannot_change_discovery_or_confirmation(self):
        initial = self.fit()
        changed = copy.deepcopy(self.parts)
        changed[1]["y"] = 1 - changed[1]["y"]
        other = self.fit(changed)
        self.assertEqual(initial["rule"], other["rule"])
        self.assertEqual(initial["matched_estimation"], other["matched_estimation"])
        self.assertEqual(initial["gate"], other["gate"])
        self.assertEqual(initial["methods"]["discovery_gated"]["delta"], other["methods"]["discovery_gated"]["delta"])
        self.assertEqual(initial["methods"]["independent_fixed"]["delta"], -other["methods"]["independent_fixed"]["delta"])
        self.assertNotEqual(initial["methods"]["independent_gated"]["delta"], other["methods"]["independent_gated"]["delta"])

    def test_gate_uses_full_discovery_sign_even_when_matched_estimate_disagrees(self):
        parts = copy.deepcopy(self.parts)
        parts[1] = sample("E", 16, 2)
        for part in parts:
            part["y"] = np.ones(len(part["p"]), dtype=int)

        def fixed_rule(x, y, p, ids, settings, seed):
            return {"bin": 2, "conditions": [], "discovery_tasks": len(np.unique(ids)),
                    "discovery_candidates": len(ids), "discovery_residual": float((y-p).mean()), "priority": 1.0}

        with patch("research.confirmation.find_rule", side_effect=fixed_rule):
            first = self.fit(parts)
            selected = first["matched_estimation"]["discovery_ids"]
            parts[0]["y"][np.isin(parts[0]["ids"], selected)] = 0
            fitted = self.fit(parts)
        self.assertGreater(fitted["rule"]["discovery_residual"], 0)
        self.assertLess(fitted["methods"]["discovery_fixed"]["estimate"], 0)
        self.assertGreater(fitted["gate"]["interval"]["estimate"], 0)
        self.assertTrue(fitted["gate"]["accepted"])

    def test_matching_records_row_count_differences_without_creating_extra_tasks(self):
        initial = self.fit()
        changed = copy.deepcopy(self.parts)
        changed[1] = {key: np.repeat(value, 3, axis=0) for key, value in changed[1].items()}
        fitted = self.fit(changed)
        self.assertEqual(fitted["matched_estimation"]["tasks"], 20)
        self.assertEqual(fitted["matched_estimation"]["discovery_candidates"], 40)
        self.assertEqual(fitted["matched_estimation"]["estimation_candidates"], 120)
        self.assertEqual(fitted["methods"]["independent_fixed"]["label_tasks"]["estimation"], 20)
        self.assertEqual(fitted["methods"]["independent_fixed"]["estimate"],
                         initial["methods"]["independent_fixed"]["estimate"])

    def test_validation_outcomes_cannot_change_factorial_corrections(self):
        initial = self.fit()
        changed = copy.deepcopy(self.parts)
        changed[3]["y"] = 1 - changed[3]["y"]
        other = self.fit(changed)
        for name in FACTORIAL:
            for key in ("rule", "delta", "step", "estimate", "eligible", "gate"):
                self.assertEqual(initial["methods"][name][key], other["methods"][name][key])
        self.assertNotEqual(initial["methods"]["discovery_fixed"]["validation_brier"],
                            other["methods"]["discovery_fixed"]["validation_brier"])

    def test_validation_can_select_zero_step_when_all_corrections_hurt(self):
        changed = copy.deepcopy(self.parts)
        changed[3]["y"] = 1 - changed[3]["y"]
        fitted = self.fit(changed)
        for name in ("discovery_shrinkage", "independent_shrinkage", "pooled_shrinkage"):
            self.assertEqual(fitted["methods"][name]["step"], 0)
            self.assertEqual(fitted["methods"][name]["delta"], 0)

    def test_support_counts_tasks_instead_of_duplicate_rows(self):
        changed = copy.deepcopy(self.parts)
        changed[1]["ids"] = np.repeat("E:one", len(changed[1]["ids"]))
        fitted = self.fit(changed)
        self.assertEqual(fitted["support"]["estimation"], 1)
        self.assertFalse(fitted["eligible"])
        self.assertIsNone(fitted["gate"]["interval"])
        for name in (*FACTORIAL, "discovery_shrinkage", "independent_shrinkage"):
            self.assertFalse(fitted["methods"][name]["eligible"])
            self.assertEqual(fitted["methods"][name]["delta"], 0)
            self.assertEqual(fitted["methods"][name]["label_tasks"]["estimation"], 0)

    def test_no_discovery_rule_leaves_all_shared_rule_methods_unchanged(self):
        changed = copy.deepcopy(self.parts)
        changed[0]["p"] = changed[0]["y"].astype(float)
        fitted = self.fit(changed)
        self.assertEqual(fitted["status"], "no_discovered_rule")
        self.assertIsNone(fitted["rule"])
        for name in (*FACTORIAL, "discovery_shrinkage", "independent_shrinkage"):
            self.assertIsNone(fitted["methods"][name]["rule"])
            self.assertEqual(fitted["methods"][name]["delta"], 0)

    def test_application_is_static_label_free_and_clipped(self):
        p = np.array([.5, .65])
        fitted = {"bins": 5, "methods": {
            "up": {"rule": {"bin": 2, "conditions": []}, "delta": .7},
            "down": {"rule": {"bin": 2, "conditions": []}, "delta": -.7},
        }}
        scores = apply_confirmation(np.zeros((2, 16)), p, fitted)
        np.testing.assert_array_equal(scores["up"], [1., .65])
        np.testing.assert_array_equal(scores["down"], [0., .65])
        np.testing.assert_array_equal(p, [.5, .65])
        self.assertEqual(list(inspect.signature(apply_confirmation).parameters), ["x", "p", "fitted"])

    def test_label_accounting_distinguishes_fit_queries_and_diagnostics(self):
        fitted = self.fit()
        self.assertEqual(fitted["available_label_tasks"],
                         {"discovery": 80, "estimation": 40, "confirmation": 40, "validation": 40, "total": 200})
        expected = {"no_update": 0, "discovery_fixed": 80, "independent_fixed": 100,
                    "discovery_gated": 100, "independent_gated": 120,
                    "discovery_shrinkage": 120, "independent_shrinkage": 140, "pooled_shrinkage": 200}
        for name, total in expected.items():
            method = fitted["methods"][name]
            self.assertEqual(method["label_tasks"]["total"], total)
            self.assertEqual(method["actual_label_tasks"]["validation"], 40)
            self.assertEqual(method["actual_label_tasks"]["total"], total + (40 if method["label_tasks"]["validation"] == 0 else 0))

    def test_fit_ignores_truth_and_has_no_test_input(self):
        class NoTruthAccess(dict):
            def __getitem__(self, key):
                if key == "truth":
                    raise AssertionError("Fitting must not read oracle truth")
                return super().__getitem__(key)

        parts = [NoTruthAccess({**part, "truth": object()}) for part in self.parts]
        fitted = self.fit(parts)
        self.assertTrue(fitted["eligible"])
        self.assertEqual(list(inspect.signature(fit_confirmation).parameters),
                         ["discovery", "estimation", "confirmation", "validation", "settings", "seed"])

    def test_overlapping_task_roles_are_rejected(self):
        changed = copy.deepcopy(self.parts)
        changed[1]["ids"][0] = changed[0]["ids"][0]
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            self.fit(changed)


if __name__ == "__main__":
    unittest.main()
