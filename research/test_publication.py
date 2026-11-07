from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest

import nbformat

from .reference import load_reference

PROJECT = Path(__file__).resolve().parents[1]


class PublicationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = load_reference()

    def test_saved_headline_metrics_match_the_reference(self):
        with (PROJECT / "results/metrics.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 36)
        for row in rows:
            expected = self.reference["summary"]["partitions"][row["split"]]["methods"][row["method"]]
            for metric in ("brier", "log_loss"):
                self.assertAlmostEqual(float(row[metric]), expected[metric], places=12)
            self.assertAlmostEqual(float(row["success"]), expected["routing"]["success"], places=12)

    def test_notebooks_are_portable_and_have_saved_successful_outputs(self):
        for relative in ("analysis.ipynb", "reconstruction/cursor-confidence-walkthrough.ipynb"):
            with self.subTest(notebook=relative):
                path = PROJECT / relative
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                self.assertEqual(notebook.metadata.kernelspec.name, "python3")
                code = [cell for cell in notebook.cells if cell.cell_type == "code"]
                self.assertTrue(all(cell.execution_count is not None for cell in code))
                self.assertFalse(any(output.output_type == "error" for cell in code for output in cell.outputs))
                self.assertNotIn("/Users/", path.read_text())

    def test_recorded_reproduction_checks_report_actual_differences(self):
        for mode in ("replay", "refit"):
            report = json.loads((PROJECT / f"reproduction/2026-11-06-{mode}.json").read_text())
            self.assertEqual(report["reference_run"], self.reference["manifest"]["run_id"])
            self.assertEqual(len(report["reference_comparison"]), 36)
            self.assertTrue(report["all_probabilities_match"])
            self.assertTrue(all(row["within_1e-8"] for row in report["reference_comparison"]))
            self.assertLess(max(row["maximum_probability_difference"] for row in report["reference_comparison"]), 1e-8)
            self.assertEqual(set(report["maximum_metric_differences"]), {"brier", "log_loss", "ece", "worst"})


if __name__ == "__main__":
    unittest.main()
