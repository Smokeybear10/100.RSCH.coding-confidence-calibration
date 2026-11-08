from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import confirmation_study as study
from .confirmation import METHODS
from .data import ROOT, digest


class StudyDesignChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(study.PROTOCOL.read_text())

    def test_source_roles_are_disjoint_and_budgets_are_nested_by_task(self):
        ids = np.repeat([f"source:{index}" for index in range(1055)], 10)
        data = {"ids": ids, "x": np.zeros((len(ids), 16)), "y": np.zeros(len(ids)),
                "p": np.full(len(ids), .5)}
        plans = study.source_plans(data, self.protocol)
        self.assertEqual(len(plans), self.protocol["source"]["outer_folds"])
        held_out = []
        for plan in plans:
            sets = [set(plan[role]) for role in ("train", "score_calibration", "test")]
            sets.extend(set(values) for values in plan["audit_roles"].values())
            self.assertEqual(set.union(*sets), set(ids))
            self.assertTrue(all(not left & right for index, left in enumerate(sets) for right in sets[index + 1:]))
            held_out.extend(plan["test"])
            previous = {role: set() for role in study.ROLES}
            for budget in self.protocol["source"]["budgets"]:
                parts = study.budget_parts(data, plan["audit_roles"], budget, self.protocol["budget_fractions"])
                counts = study.role_counts(budget, self.protocol["budget_fractions"])
                self.assertEqual(sum(counts.values()), budget)
                for role, part in parts.items():
                    tasks = set(part["ids"])
                    self.assertEqual(len(tasks), counts[role])
                    self.assertTrue(previous[role] <= tasks)
                    self.assertEqual(len(part["ids"]), 10 * len(tasks))
                    self.assertFalse(tasks & set(plan["test"]))
                    previous[role] = tasks
        self.assertEqual(len(held_out), 1055)
        self.assertEqual(len(set(held_out)), 1055)

    def test_budget_overflow_cannot_reuse_tasks(self):
        fractions = self.protocol["budget_fractions"]
        with self.assertRaisesRegex(ValueError, "Not enough independent tasks"):
            study.role_ids(np.array(["one", "two"]), 3, fractions, 17)
        masters = study.role_ids(np.array([str(index) for index in range(120)]), 120, fractions, 17)
        with self.assertRaisesRegex(ValueError, "exceeds its master role"):
            study.budget_parts({"ids": np.array(["one"])}, masters, 240, fractions)

    def test_shared_and_independent_labels_preserve_the_same_probability_problem(self):
        scenario = {"effect": .25, "prevalence": .25}
        independent = study.synthetic_data(128, 12345, scenario, False)
        shared = study.synthetic_data(128, 12345, scenario, True)
        null = study.synthetic_data(128, 12345, {"effect": 0, "prevalence": .25}, False)
        for name in ("x", "p", "truth", "ids"):
            np.testing.assert_array_equal(independent[name], shared[name])
        np.testing.assert_array_equal(null["p"], null["truth"])
        np.testing.assert_array_equal(null["x"], shared["x"])
        self.assertTrue(np.any(shared["truth"] < shared["p"]))
        self.assertFalse(np.array_equal(independent["y"], shared["y"]))
        shared_monotone, independent_monotone = [], []
        for task in np.unique(shared["ids"]):
            mask = shared["ids"] == task
            order = np.argsort(shared["truth"][mask])
            shared_monotone.append(np.all(np.diff(shared["y"][mask][order]) >= 0))
            independent_monotone.append(np.all(np.diff(independent["y"][mask][order]) >= 0))
        self.assertTrue(all(shared_monotone))
        self.assertFalse(all(independent_monotone))

    def test_final_seeds_are_unique_and_exclude_pilots(self):
        seeds = self.protocol["simulation"]["seeds"]
        self.assertEqual(len(seeds), 100)
        self.assertEqual(len(set(seeds)), 100)
        self.assertFalse(set(seeds) & set(study.PILOT_SEEDS))

    def test_exact_binomial_intervals_keep_boundary_uncertainty(self):
        low, high = study.binomial_interval(0, 100)
        self.assertEqual(low, 0)
        self.assertAlmostEqual(high, 1 - .025 ** .01, places=12)
        low, high = study.binomial_interval(100, 100)
        self.assertAlmostEqual(low, .025 ** .01, places=12)
        self.assertEqual(high, 1)
        first = study.binomial_interval(25, 100)
        reflected = study.binomial_interval(75, 100)
        self.assertAlmostEqual(first[0], 1 - reflected[1], places=12)
        self.assertAlmostEqual(first[1], 1 - reflected[0], places=12)
        for counts in [(0, 0), (-1, 10), (11, 10)]:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                study.binomial_interval(*counts)

    def test_source_contrasts_pair_task_losses_before_bootstrapping(self):
        rows = [{"budget": 120, "task": f"task:{task}", "method": method,
                 "brier": .3 + index * .01 + task * .001}
                for task in range(3) for index, method in enumerate(METHODS)]
        frame = pd.DataFrame(rows)
        paired = study.contrasts(frame, ["budget"], "task", "brier", self.protocol)
        primary = paired.loc[paired["contrast"] == "gate_with_independent_estimate"].iloc[0]
        self.assertEqual(primary["task_clusters"], 3)
        for field in ("difference", "low", "high"):
            self.assertAlmostEqual(primary[field], .02, places=12)
        interaction = paired.loc[paired["contrast"] == "gate_estimator_interaction"].iloc[0]
        self.assertAlmostEqual(interaction["difference"], 0, places=12)
        self.assertEqual(len(paired), len(self.protocol["contrasts"]) + 1)
        with self.assertRaisesRegex(ValueError, "missing a method or replication"):
            study.contrasts(frame.iloc[1:], ["budget"], "task", "brier", self.protocol)
        with self.assertRaises(ValueError):
            study.contrasts(pd.concat([frame, frame.iloc[:1]]), ["budget"], "task", "brier", self.protocol)


class ExternalFreezeChecks(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(study.PROTOCOL.read_text())
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.models = [{"fold": fold, "budgets": {str(self.protocol["source"]["external_primary_budget"]): {}}}
                       for fold in range(self.protocol["source"]["outer_folds"])]
        (self.output / "protocol.json").write_text(json.dumps(self.protocol))
        self.write_models(self.models)
        self.seal = {"protocol_sha256": digest(self.output / "protocol.json"),
                     "fitted_models_sha256": digest(self.output / "fitted_models.json"),
                     "code_sha256": {f"research/{name}": digest(ROOT / name) for name in study.CORE_FILES},
                     "external_outcomes_loaded_by_driver": False}
        self.write_seal()

    def write_models(self, models: list[dict]) -> None:
        (self.output / "fitted_models.json").write_text(json.dumps(models))

    def write_seal(self) -> None:
        (self.output / "source_freeze.json").write_text(json.dumps(self.seal))

    def reject_before_loading(self, message: str, protocol: dict | None = None) -> None:
        with patch("research.external.load_external", side_effect=AssertionError("External outcomes must stay unread")) as loader:
            with self.assertRaisesRegex(ValueError, message):
                study.external_experiment(self.output / "unread-external", self.protocol if protocol is None else protocol, self.output)
            loader.assert_not_called()

    def test_changed_model_file_is_rejected_before_external_outcomes_load(self):
        with (self.output / "fitted_models.json").open("a") as handle:
            handle.write(" ")
        self.reject_before_loading("Frozen source models changed")

    def test_changed_protocol_file_or_object_is_rejected_before_loading(self):
        changed = copy.deepcopy(self.protocol)
        changed["question"] = "Changed after the source fit"
        self.reject_before_loading("Frozen protocol changed", changed)
        with (self.output / "protocol.json").open("a") as handle:
            handle.write(" ")
        self.reject_before_loading("Frozen protocol changed")

    def test_changed_implementation_seal_is_rejected_before_loading(self):
        self.seal["code_sha256"][next(iter(self.seal["code_sha256"]))] = "changed"
        self.write_seal()
        self.reject_before_loading("implementation changed")

    def test_every_frozen_fold_and_primary_budget_are_required(self):
        incomplete = self.models[:-1]
        duplicate = copy.deepcopy(self.models)
        duplicate[-1]["fold"] = duplicate[0]["fold"]
        wrong_budget = copy.deepcopy(self.models)
        wrong_budget[0]["budgets"] = {}
        for models in (incomplete, duplicate, wrong_budget):
            with self.subTest(models=models):
                self.write_models(models)
                self.seal["fitted_models_sha256"] = digest(self.output / "fitted_models.json")
                self.write_seal()
                self.reject_before_loading("every frozen source fold and the primary budget")

    def test_valid_seal_reaches_only_the_mock_loader(self):
        class LoaderReached(RuntimeError):
            pass

        with patch("research.external.load_external", side_effect=LoaderReached("Expected preflight completion")) as loader:
            with self.assertRaises(LoaderReached):
                study.external_experiment(self.output / "unread-external", self.protocol, self.output)
            loader.assert_called_once_with(self.output / "unread-external")

    def test_smoke_with_external_refuses_before_creating_output_or_loading_data(self):
        output = self.output / "must-not-be-created"
        with patch("research.external.load_external", side_effect=AssertionError("External outcomes must stay unread")) as loader:
            with self.assertRaisesRegex(ValueError, "Smoke runs never evaluate"):
                study.run_study(output, self.output / "missing-protocol.json", self.output / "unread-external", smoke=True)
            loader.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
