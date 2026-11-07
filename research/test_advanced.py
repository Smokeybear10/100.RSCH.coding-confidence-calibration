from __future__ import annotations

import copy
import json
import unittest

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from .advanced import archive_data, fit_fold, make_parts, optimal_validation_cutoff, routing_tables, sample_rows, task_folds
from .auditors import apply_auditor, fit_auditor, leaf_rules, residual_interval, rule_mask
from .data import ROOT
from .methods import budget_threshold, route_mask
from .reference import load_reference
from .simulations import synthetic_tasks


class AdvancedProtocolChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = load_reference()
        cls.data = archive_data(cls.reference)
        cls.protocol = json.loads((ROOT / "advanced_protocol.json").read_text())
        cls.plans = task_folds(cls.data, cls.protocol)
        cls.plan = cls.plans[0]
        cls.groups = cls.reference["protocol"]["groups"]
        cls.fitted = fit_fold(cls.data, cls.plan, cls.protocol, cls.groups)

    def test_every_task_is_evaluated_once_per_repeat_and_roles_are_disjoint(self):
        all_ids = set(self.data["ids"])
        for seed in self.protocol["seeds"]:
            held_out = []
            for plan in [value for value in self.plans if value["seed"] == seed]:
                sets = [set(ids) for ids in plan["roles"].values()]
                self.assertEqual(set.union(*sets), all_ids)
                self.assertTrue(all(not left.intersection(right) for i, left in enumerate(sets) for right in sets[i+1:]))
                held_out.extend(plan["roles"]["test"])
                parts = make_parts(self.data, plan)
                self.assertTrue(all(len(part["y"]) == 10*len(np.unique(part["ids"])) for part in parts.values()))
            self.assertEqual(len(held_out), len(all_ids))
            self.assertEqual(set(held_out), all_ids)

    def test_test_and_validation_labels_do_not_change_fitted_predictions(self):
        changed = {name: values.copy() for name, values in self.data.items()}
        mask = np.isin(changed["ids"], self.plan["roles"]["test"]+self.plan["roles"]["validation"])
        changed["y"][mask] = 1-changed["y"][mask]
        changed["alternate"][mask] = 1-changed["alternate"][mask]
        other = fit_fold(changed, self.plan, self.protocol, self.groups)
        self.assertEqual(other["details"], self.fitted["details"])
        for method, p in self.fitted["predictions"]["test"].items():
            np.testing.assert_array_equal(p, other["predictions"]["test"][method])
        np.testing.assert_array_equal(self.fitted["benefit"]["test"], other["benefit"]["test"])

    def test_all_variants_are_bounded_and_confirmation_blocks_do_not_overlap(self):
        self.assertEqual(len(self.fitted["predictions"]["test"]), 10)
        for methods in self.fitted["predictions"].values():
            for p in methods.values():
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(((p >= 0) & (p <= 1)).all())
        allowed = set(self.plan["roles"]["confirmation"])
        for model in self.fitted["details"].values():
            blocks = model["calibrators"]["confirmed_tree"]["confirmation_blocks"]
            flattened = [task for block in blocks for task in block]
            self.assertEqual(set(flattened), allowed)
            self.assertEqual(len(flattened), len(set(flattened)))

    def test_full_sample_learning_curve_reproduces_full_auditor_fit(self):
        rows = sample_rows(self.fitted, self.protocol, self.groups, self.plan["seed"]+self.plan["fold"])
        largest = max(row["auditor_tasks"] for row in rows)
        test = self.fitted["parts"]["test"]
        for row in [row for row in rows if row["auditor_tasks"] == largest]:
            expected = np.mean((self.fitted["predictions"]["test"]["logistic/"+row["method"]]-test["y"])**2)
            self.assertAlmostEqual(row["test_brier"], expected, places=12)

    def test_policy_cutoffs_do_not_use_test_outcomes_and_task_costs_match_totals(self):
        _, costs, task_costs = routing_tables(self.fitted, self.protocol)
        self.assertTrue(all(row["mean_compute"] == 4 for row in costs if row["method"] == "alternate_direct"))
        changed = {**self.fitted, "parts": {**self.fitted["parts"]}}
        test = self.fitted["parts"]["test"]
        changed["parts"]["test"] = {**test, "y": 1-test["y"], "alternate": 1-test["alternate"]}
        _, other, _ = routing_tables(changed, self.protocol)
        keys = ("method", "selector", "failure_loss", "validation_budget", "cutoff_score", "cutoff_tie_key")
        self.assertEqual([[row[key] for key in keys] for row in costs], [[row[key] for key in keys] for row in other])
        for row in costs:
            values = [value["objective"] for value in task_costs
                      if all(value[key] == row[key] for key in ("method", "selector", "failure_loss"))]
            self.assertAlmostEqual(float(np.mean(values)), row["total_objective"], places=12)


class AuditorChecks(unittest.TestCase):
    def setUp(self):
        protocol = json.loads((ROOT / "advanced_protocol.json").read_text())
        self.settings = {**protocol["auditor"], "min_samples_leaf": 10,
                         "min_discovery_tasks": 10, "min_confirmation_tasks": 4}

    def test_serialized_tree_paths_match_sklearn_float32_regions(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(200, 16))
        tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=10).fit(x, (x[:, 1] > 0).astype(float)+x[:, 2])
        memberships = []
        for conditions in leaf_rules(tree):
            mask = rule_mask(x, np.repeat(.1, len(x)), {"bin": 0, "conditions": conditions}, 5)
            self.assertEqual(len(np.unique(tree.apply(x)[mask])), 1)
            memberships.append(mask)
        np.testing.assert_array_equal(np.sum(memberships, axis=0), np.ones(len(x)))

    def test_future_confirmation_labels_cannot_change_the_first_decision(self):
        rng = np.random.default_rng(19)
        discovery_x, confirmation_x = rng.normal(size=(500, 16)), rng.normal(size=(300, 16))
        discovery = {"x": discovery_x, "p": np.repeat(.8, 500), "y": (discovery_x[:, 1] < 0).astype(int),
                     "ids": np.arange(500).astype(str)}
        confirmation = {"x": confirmation_x, "p": np.repeat(.8, 300), "y": (confirmation_x[:, 1] < 0).astype(int),
                        "ids": np.arange(1000, 1300).astype(str)}
        fitted = fit_auditor(discovery, confirmation, self.settings, 11)
        self.assertTrue(fitted["trace"][0]["accepted"])
        changed = copy.deepcopy(confirmation)
        future = ~np.isin(changed["ids"], fitted["confirmation_blocks"][0])
        changed["y"][future] = 1-changed["y"][future]
        other = fit_auditor(discovery, changed, self.settings, 11)
        self.assertEqual(fitted["trace"][0], other["trace"][0])

    def test_task_bootstrap_does_not_treat_duplicate_answers_as_new_tasks(self):
        values, ids = np.arange(20)/20-.5, np.arange(20).astype(str)
        first = residual_interval(values, ids, 123, 500, .05)
        repeated = residual_interval(np.repeat(values, 10), np.repeat(ids, 10), 123, 500, .05)
        np.testing.assert_allclose([first["low"], first["high"]], [repeated["low"], repeated["high"]])

    def test_replay_uses_current_score_bands_without_labels(self):
        x = np.zeros((2, 16))
        fitted = {"bins": 5, "patches": [{"bin": 4, "conditions": [], "delta": -.3},
                                          {"bin": 2, "conditions": [], "delta": -.8}]}
        np.testing.assert_allclose(apply_auditor(x, np.array([.82, 1.]), fitted), [0, .7])

    def test_benefit_ranking_handles_zero_and_full_budgets(self):
        benefits = np.linspace(-1, 1, 101)
        ranking, keys = (1-benefits)/2, np.arange(101)/101
        self.assertFalse(route_mask(ranking, budget_threshold(ranking, 0, keys), keys).any())
        self.assertTrue(route_mask(ranking, budget_threshold(ranking, 1, keys), keys).all())

    def test_exact_validation_search_handles_ties_and_direct_generation_cost(self):
        scores = np.array([.1, .1, .4, .9])
        part = {"y": np.array([1, 0, 0, 1]), "alternate": np.array([0, 1, 1, 1]),
                "keys": np.array([.2, .1, .3, .4])}
        settings = {"primary_cost": 1, "alternate_cost": 4}
        threshold, budget = optimal_validation_cutoff(scores, part, settings, 10)
        routed = route_mask(scores, threshold, part["keys"])
        np.testing.assert_array_equal(routed, [False, True, False, False])
        self.assertEqual(budget, .25)
        outcome = np.where(routed, part["alternate"], part["y"])
        self.assertEqual(float(np.mean(1+4*routed+10*(1-outcome))), 4.5)


class SimulationChecks(unittest.TestCase):
    def test_null_and_biased_generators_have_known_probabilities(self):
        null = synthetic_tasks(100, 123, False)
        biased = synthetic_tasks(100, 123, True)
        np.testing.assert_array_equal(null["p"], null["truth"])
        self.assertTrue(np.any(biased["truth"] < biased["p"]))
        np.testing.assert_array_equal(null["x"], biased["x"])
        self.assertEqual(len(np.unique(null["ids"])), 100)
        self.assertEqual(len(null["y"]), 1000)


if __name__ == "__main__":
    unittest.main()
