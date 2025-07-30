from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import unittest

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .confirmation import METHODS
from .confirmation_study import CORE_FILES, PILOT_SEEDS
from .data import FEATURES, ROOT, digest
from .reference import load_reference

PROJECT = ROOT.parent
SOURCE_METHODS = (*METHODS, "raw_logistic")


class ConfirmationArtifactChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = PROJECT / "confirmation_results"
        cls.summary = json.loads((cls.directory / "summary.json").read_text())
        cls.protocol = json.loads((cls.directory / "protocol.json").read_text())
        cls.manifest = json.loads((cls.directory / "manifest.json").read_text())
        cls.reference = load_reference()
        cls.source = pd.read_csv(cls.directory / "source_predictions.csv.gz", float_precision="round_trip")
        cls.source_losses = pd.read_csv(cls.directory / "source_task_losses.csv.gz", float_precision="round_trip")
        cls.simulations = pd.read_csv(cls.directory / "simulation_runs.csv")

    def assert_table_matches(self, computed: pd.DataFrame, saved: pd.DataFrame,
                             keys: list[str], columns: list[str]) -> None:
        joined = computed.merge(saved[keys + columns], on=keys, how="outer",
                                validate="one_to_one", suffixes=("", "_saved"), indicator=True)
        self.assertTrue(joined["_merge"].eq("both").all())
        for column in columns:
            np.testing.assert_allclose(joined[column], joined[column + "_saved"],
                                       rtol=0, atol=1e-12, equal_nan=True)

    def assert_contrast_means(self, units: pd.DataFrame, saved: pd.DataFrame,
                              keys: list[str], unit: str, metric: str) -> None:
        computed = []
        groups = units.groupby(keys) if keys else [((), units)]
        for values, frame in groups:
            values = values if isinstance(values, tuple) else (values,)
            identity = dict(zip(keys, values))
            wide = frame.pivot(index=unit, columns="method", values=metric)
            self.assertFalse(wide[list(METHODS)].isna().any().any())
            for contrast in self.protocol["contrasts"]:
                difference = float((wide[contrast["left"]] - wide[contrast["right"]]).mean())
                computed.append({**identity, "contrast": contrast["name"], "difference": difference,
                                 "units": len(wide)})
            interaction = ((wide["independent_gated"] - wide["independent_fixed"])
                           - (wide["discovery_gated"] - wide["discovery_fixed"]))
            computed.append({**identity, "contrast": "gate_estimator_interaction",
                             "difference": float(interaction.mean()), "units": len(wide)})
        column = "seeds" if unit == "seed" else "task_clusters"
        actual = saved.rename(columns={column: "units"})
        self.assert_table_matches(pd.DataFrame(computed), actual, keys + ["contrast"], ["difference", "units"])

    def test_full_run_manifest_protocol_and_numerical_sources_match(self):
        required = {"protocol.json", "source_predictions.csv.gz", "source_task_losses.csv.gz",
                    "source_metrics.csv", "source_contrasts.csv", "splits.json", "fitted_models.json",
                    "source_freeze.json", "simulation_runs.csv", "simulation_summary.csv",
                    "simulation_contrasts.csv", "simulation_fits.jsonl.gz", "external_metrics.csv",
                    "external_contrasts.csv", "external_record.json", "summary.json"}
        self.assertTrue(required <= set(self.manifest["files"]))
        for name, expected in self.manifest["files"].items():
            with self.subTest(file=name):
                self.assertEqual(Path(name).name, name)
                self.assertEqual(digest(self.directory / name), expected)
        self.assertFalse(self.summary["smoke"])
        self.assertTrue(self.summary["external_evaluated"])
        self.assertEqual(self.summary["source_folds"], 5)
        self.assertEqual(self.summary["source_folds"], self.protocol["source"]["outer_folds"])
        self.assertEqual(self.summary["simulation_seeds"], self.protocol["simulation"]["seeds"])
        self.assertEqual(len(set(self.summary["simulation_seeds"])), 100)
        self.assertFalse(set(self.summary["simulation_seeds"]) & set(PILOT_SEEDS))
        self.assertEqual(digest(self.directory / "protocol.json"), self.summary["protocol_sha256"])
        self.assertEqual(digest(ROOT / "confirmation_protocol.json"), self.summary["protocol_sha256"])
        self.assertEqual(set(self.summary["code_sha256"]), {f"research/{name}" for name in CORE_FILES})
        for name in CORE_FILES:
            self.assertEqual(digest(ROOT / name), self.summary["code_sha256"][f"research/{name}"])

    def test_source_predictions_preserve_original_labels_and_task_identity(self):
        source = self.source
        reference = pd.DataFrame([
            {"task": row["task_id"], "candidate": row["candidate"], "expected_y": row["label"],
             "expected_empty": row["features"][FEATURES.index("empty_code")]}
            for row in self.reference["rows"]
        ])
        joined = source.merge(reference, on=["task", "candidate"], how="left", validate="many_to_one")
        np.testing.assert_array_equal(joined["y"], joined["expected_y"])
        np.testing.assert_array_equal(joined["empty_code"], joined["expected_empty"])
        self.assertEqual(set(source["method"]), set(SOURCE_METHODS))
        self.assertEqual(set(source["budget"]), set(self.protocol["source"]["budgets"]))
        self.assertEqual(set(source["fold"]), set(range(self.protocol["source"]["outer_folds"])))
        expected = len(reference) * len(SOURCE_METHODS) * len(self.protocol["source"]["budgets"])
        self.assertEqual(len(source), expected)
        self.assertFalse(source.duplicated(["budget", "method", "task", "candidate"]).any())
        self.assertTrue(source.groupby(["budget", "method", "task"]).size().eq(10).all())
        self.assertTrue(source["candidate"].isin(range(10)).all())
        self.assertTrue(np.isfinite(source[["p", "initial_p"]]).all().all())
        self.assertTrue(source["p"].between(0, 1).all())
        self.assertTrue(source["initial_p"].between(0, 1).all())
        task_count = reference["task"].nunique()
        self.assertTrue(source.groupby(["budget", "method"])["task"].nunique().eq(task_count).all())
        self.assertEqual(task_count, self.summary["source_task_clusters"])
        plans = json.loads((self.directory / "splits.json").read_text())
        folds = {task: plan["fold"] for plan in plans for task in plan["test"]}
        self.assertEqual(len(folds), task_count)
        self.assertTrue(source["task"].map(folds).eq(source["fold"]).all())
        baseline = source.loc[source["method"] == "no_update"]
        np.testing.assert_array_equal(baseline["p"], baseline["initial_p"])

    def test_source_loss_tables_and_contrasts_recompute_from_saved_predictions(self):
        frame = self.source.copy()
        frame["calculated_brier"] = (frame["p"] - frame["y"]) ** 2
        bounded = frame["p"].clip(1e-6, 1 - 1e-6)
        frame["calculated_log_loss"] = -frame["y"] * np.log(bounded) - (1 - frame["y"]) * np.log(1 - bounded)
        np.testing.assert_allclose(frame["brier"], frame["calculated_brier"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(frame["log_loss"], frame["calculated_log_loss"], rtol=0, atol=1e-12)
        keys = ["fold", "budget", "method"]
        measured = frame.groupby(keys, as_index=False).agg(
            brier=("calculated_brier", "mean"), log_loss=("calculated_log_loss", "mean"),
            tasks=("task", "nunique"), candidates=("task", "size"),
        )
        nonempty = frame.loc[frame["empty_code"] == 0].groupby(keys, as_index=False).agg(
            nonempty_brier=("calculated_brier", "mean"), nonempty_candidates=("task", "size"),
        )
        measured = measured.merge(nonempty, on=keys, how="left", validate="one_to_one")
        measured["nonempty_candidates"] = measured["nonempty_candidates"].fillna(0)
        saved = pd.read_csv(self.directory / "source_metrics.csv")
        self.assert_table_matches(measured, saved, keys,
                                  ["brier", "log_loss", "nonempty_brier", "nonempty_candidates", "tasks", "candidates"])
        task_keys = ["budget", "method", "task"]
        losses = frame.groupby(task_keys, as_index=False).agg(
            brier=("calculated_brier", "mean"), log_loss=("calculated_log_loss", "mean"),
        )
        self.assert_table_matches(losses, self.source_losses, task_keys, ["brier", "log_loss"])
        contrasts = pd.read_csv(self.directory / "source_contrasts.csv")
        self.assert_contrast_means(losses, contrasts, ["budget"], "task", "brier")
        experimental = saved.loc[saved["method"].isin(METHODS)]
        self.assertTrue(experimental["available_label_tasks"].eq(experimental["budget"]).all())
        self.assertTrue(experimental["queried_label_tasks"].ge(0).all())
        self.assertTrue(experimental["queried_label_tasks"].le(experimental["queried_with_diagnostics"]).all())
        self.assertTrue(experimental["queried_with_diagnostics"].le(experimental["available_label_tasks"]).all())

    def test_simulation_grid_has_complete_independent_seed_pairs_and_known_null(self):
        frame = self.simulations
        config = self.protocol["simulation"]
        keys = ["seed", "scenario", "shared_uniform", "budget", "method"]
        expected = pd.MultiIndex.from_product([
            config["seeds"], [scenario["name"] for scenario in config["scenarios"]],
            config["shared_label_uniform"], config["budgets"], METHODS,
        ], names=keys)
        actual = pd.MultiIndex.from_frame(frame[keys])
        self.assertEqual(len(frame), 19200)
        self.assertFalse(actual.has_duplicates)
        self.assertEqual(set(actual), set(expected))
        self.assertTrue(frame.groupby(keys[1:])["seed"].nunique().eq(100).all())
        self.assertTrue(frame["tasks"].eq(config["test_tasks"]).all())
        self.assertTrue(frame["candidates"].eq(config["test_tasks"] * config["candidates_per_task"]).all())
        self.assertTrue(np.isfinite(frame[["known_probability_mse", "initial_probability_mse", "brier", "log_loss"]]).all().all())
        self.assertTrue(frame["known_probability_mse"].between(0, 1).all())
        self.assertTrue(frame["brier"].between(0, 1).all())
        baseline = frame.loc[frame["method"] == "no_update"]
        np.testing.assert_array_equal(baseline["known_probability_mse"], baseline["initial_probability_mse"])
        self.assertFalse(baseline["any_update"].any())
        null = baseline.loc[baseline["scenario"] == "calibrated_null"]
        self.assertTrue(null["known_probability_mse"].eq(0).all())
        self.assertTrue(frame["available_label_tasks"].eq(frame["budget"]).all())
        contrasts = pd.read_csv(self.directory / "simulation_contrasts.csv")
        self.assert_contrast_means(frame, contrasts, ["scenario", "shared_uniform", "budget"],
                                   "seed", "known_probability_mse")

    def test_simulation_summaries_and_update_intervals_match_seed_records(self):
        rows = []
        keys = ["scenario", "shared_uniform", "budget", "method"]
        for identity, group in self.simulations.groupby(keys):
            updates = int(group["any_update"].sum())
            exact = binomtest(updates, len(group)).proportion_ci(confidence_level=.95, method="exact")
            eligible = group.loc[group["proposal_eligible"]]
            signal = group.loc[group["proposal_eligible"] & group["selected_rule_signal"]]
            gated = bool(group["policy_uses_gate"].iloc[0])
            self.assertEqual(group["policy_uses_gate"].nunique(), 1)
            rows.append({**dict(zip(keys, identity)),
                         "known_probability_mse": group["known_probability_mse"].mean(),
                         "known_probability_mse_se": group["known_probability_mse"].std(ddof=1) / np.sqrt(len(group)),
                         "brier": group["brier"].mean(), "update_rate": updates / len(group),
                         "update_low": exact.low, "update_high": exact.high,
                         "harmful_update_rate": group["harmful_update"].mean(),
                         "eligibility_rate": group["eligible"].mean(),
                         "common_discovery_rate": group["proposal_found"].mean(),
                         "common_proposal_eligibility_rate": group["proposal_eligible"].mean(),
                         "gate_acceptance_rate": group["gate_accepted"].mean() if gated else np.nan,
                         "gate_acceptance_when_eligible": eligible["gate_accepted"].mean() if gated and len(eligible) else np.nan,
                         "gate_acceptance_when_signal": signal["gate_accepted"].mean() if gated and len(signal) else np.nan,
                         "eligible_signal_seeds": len(signal) if gated else np.nan, "seeds": len(group)})
        computed = pd.DataFrame(rows)
        saved = pd.read_csv(self.directory / "simulation_summary.csv")
        self.assert_table_matches(computed, saved, keys, [name for name in computed.columns if name not in keys])
        nongated = saved.loc[~saved["method"].isin(["discovery_gated", "independent_gated"])]
        self.assertTrue(nongated[["gate_acceptance_rate", "gate_acceptance_when_eligible", "gate_acceptance_when_signal"]].isna().all().all())

    def test_external_aggregates_use_the_sealed_five_fold_ensemble(self):
        record = json.loads((self.directory / "external_record.json").read_text())
        seal = json.loads((self.directory / "source_freeze.json").read_text())
        models = json.loads((self.directory / "fitted_models.json").read_text())
        self.assertEqual(record["tasks"], 161)
        self.assertEqual(record["candidates"], 1610)
        self.assertEqual(record["source_folds_averaged"], 5)
        self.assertEqual(record["source_budget"], self.protocol["source"]["external_primary_budget"])
        self.assertEqual(record["source_budget"], 339)
        self.assertEqual(record["source"], self.protocol["external"])
        self.assertEqual(len(models), 5)
        self.assertEqual({model["fold"] for model in models}, set(range(5)))
        self.assertTrue(all("339" in model["budgets"] for model in models))
        self.assertEqual(seal["fitted_models_sha256"], digest(self.directory / "fitted_models.json"))
        self.assertEqual(seal["protocol_sha256"], self.summary["protocol_sha256"])
        self.assertEqual(seal["code_sha256"], self.summary["code_sha256"])
        self.assertFalse(seal["external_outcomes_loaded_by_driver"])
        self.assertEqual(record["source_freeze_sha256"], digest(self.directory / "source_freeze.json"))
        self.assertLess(datetime.fromisoformat(seal["frozen_at"]), datetime.fromisoformat(record["evaluated_at"]))
        self.assertLessEqual(datetime.fromisoformat(record["evaluated_at"]), datetime.fromisoformat(self.summary["created_at"]))
        for field in ("extraction_manifest_sha256", "extraction_rows_sha256"):
            self.assertIsNotNone(re.fullmatch("[0-9a-f]{64}", record[field]))
        metrics = pd.read_csv(self.directory / "external_metrics.csv")
        self.assertEqual(len(metrics), 9)
        self.assertEqual(set(metrics["method"]), set(SOURCE_METHODS))
        self.assertTrue(metrics["tasks"].eq(161).all())
        self.assertTrue(metrics["candidates"].eq(1610).all())
        self.assertTrue(np.isfinite(metrics[["brier", "log_loss"]]).all().all())
        self.assertFalse(metrics.loc[metrics["method"].isin(["raw_logistic", "no_update"]), "any_update"].any())
        contrasts = pd.read_csv(self.directory / "external_contrasts.csv")
        self.assertEqual(len(contrasts), len(self.protocol["contrasts"]) + 1)
        self.assertTrue(contrasts["task_clusters"].eq(161).all())
        means = metrics.set_index("method")["brier"]
        for contrast in self.protocol["contrasts"]:
            saved = contrasts.loc[contrasts["contrast"] == contrast["name"], "difference"].item()
            self.assertAlmostEqual(saved, means[contrast["left"]] - means[contrast["right"]], places=12)
        interaction = ((means["independent_gated"] - means["independent_fixed"])
                       - (means["discovery_gated"] - means["discovery_fixed"]))
        self.assertAlmostEqual(contrasts.loc[contrasts["contrast"] == "gate_estimator_interaction", "difference"].item(), interaction, places=12)

    def test_external_row_level_material_stays_out_of_publication(self):
        files = {path.name for path in self.directory.iterdir() if path.is_file()}
        external_files = {name for name in files if name.startswith("external_") and not name.endswith((".png", ".pdf", ".svg"))}
        self.assertEqual(external_files, {"external_metrics.csv", "external_contrasts.csv", "external_record.json"})
        self.assertNotIn("rows.json.gz", files)
        forbidden = {"task", "task_id", "candidate", "y", "label", "p", "features", "prompt", "program", "token_logprobs"}
        for name in ("external_metrics.csv", "external_contrasts.csv"):
            self.assertFalse(forbidden & set(pd.read_csv(self.directory / name, nrows=0).columns))
        tracked = subprocess.check_output(["git", "ls-files", "--", "work/multipl-e-python", "research/data/multipl-e_qwen3"],
                                          cwd=PROJECT, text=True)
        self.assertEqual(tracked, "")


if __name__ == "__main__":
    unittest.main()
