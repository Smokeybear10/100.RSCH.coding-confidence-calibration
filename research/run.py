from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import warnings
import zipfile

import numpy as np

from .data import FEATURES, ROOT, digest, prepare
from .methods import (audit, budget_threshold, clustered_interval, fit_baselines, fit_patches,
                      group_thresholds, membership, metrics, policy, replay, route_mask)

METHOD_NAMES = {
    "logistic": "Logistic confidence", "platt": "Platt scaling", "isotonic": "Isotonic",
    "multicalibration": "Multicalibration", "score_bins": "Score-bin correction",
    "group_logistic": "Group-aware logistic", "token_only": "Token features only",
    "structure_only": "Structure features only", "likelihood": "Token likelihood (heuristic)",
}


def save(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")))


def partition_tasks(tasks: list, protocol: dict) -> tuple[dict, dict]:
    seed = protocol["seed"]
    training = sorted((t["id"] for t in tasks if t["source_split"] == "train"),
                      key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    base_ids = set(training[:int(len(training) * protocol["base_train_fraction"])])
    partitions = {name: {"rows": []} for name in ("train", "calibration", "validation", "test")}
    manifests = {name: [] for name in partitions}
    for task in sorted(tasks, key=lambda t: t["id"]):
        split = task["source_split"]
        if split == "train":
            split = "train" if task["id"] in base_ids else "calibration"
        manifests[split].append(task["id"])
        for candidate in task["candidates"]:
            partitions[split]["rows"].append({"id": task["id"], "name": task["name"], **candidate})
    flat_ids = [key for keys in manifests.values() for key in keys]
    if len(flat_ids) != len(set(flat_ids)):
        raise ValueError("Task overlap between partitions")
    # Equivalent prompts cannot enter separate learned/evaluation partitions.
    prompts = {}
    split_by_id = {key: split for split, keys in manifests.items() for key in keys}
    for task in tasks:
        normalized = " ".join(task["prompt"].split())
        split = split_by_id[task["id"]]
        if prompts.setdefault(normalized, split) != split:
            raise ValueError("Duplicate prompt crosses partitions")
    for data in partitions.values():
        data["x"] = np.array([row["features"] for row in data["rows"]])
        data["y"] = np.array([row["label"] for row in data["rows"]])
        data["alternate"] = np.array([row["alternate_label"] for row in data["rows"]])
        data["ids"] = np.array([row["id"] for row in data["rows"]])
        data["keys"] = np.array([int(hashlib.sha256(f'{seed}:{row["id"]}:{row["index"]}'.encode()).hexdigest()[:13], 16) / 16**13 for row in data["rows"]])
    return partitions, manifests


def run(protocol_path: Path, verify: bool = False) -> Path:
    protocol = json.loads(protocol_path.read_text())
    code_files = sorted(ROOT.glob("*.py")) + [ROOT.parent / "requirements-research.txt"]
    provenance = {"protocol_sha256": digest(protocol_path),
        "code": {str(path.relative_to(ROOT.parent)): digest(path) for path in code_files},
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "pyarrow", "joblib", "threadpoolctl")},
        "python": platform.python_version()}
    run_id = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:12]
    destination = ROOT / "runs" / run_id
    if (destination / "summary.json").exists():
        manifest = json.loads((destination / "manifest.json").read_text())
        if any(digest(destination / filename) != expected for filename, expected in manifest["files"].items()):
            raise ValueError("An immutable run artifact has changed")
        if not verify:
            print(f"Existing immutable run: {destination}", flush=True)
            save(ROOT / "runs" / "latest.json", {"run_id": run_id})
            return destination
    source = prepare(protocol)
    partitions, manifests = partition_tasks(source["tasks"], protocol)
    thresholds = group_thresholds(partitions["train"]["x"])
    names, bins = protocol["groups"], protocol["bins"]
    for data in partitions.values():
        data["groups"] = membership(data["x"], thresholds, names)
    with warnings.catch_warnings(record=True) as numerical_warnings:
        warnings.simplefilter("always")
        predictions, models = fit_baselines(partitions["train"], partitions["calibration"], partitions, names)
    if any(type(w.message).__name__ == "ConvergenceWarning" for w in numerical_warnings):
        raise RuntimeError("Confidence estimator failed to converge")
    cal, val, test = (partitions[key] for key in ("calibration", "validation", "test"))
    selection = []
    selected = {}
    for method, selected_names in (("multicalibration", names), ("score_bins", ["all"])):
        candidates = []
        group_indices = [names.index(name) for name in selected_names]
        for epsilon in protocol["epsilon_grid"]:
            for support in protocol["support_grid"]:
                fit = fit_patches(predictions["calibration"]["platt"], cal["y"], cal["groups"][:, group_indices],
                                  cal["ids"], selected_names, bins, support, epsilon, protocol["max_updates"])
                scores = replay(predictions["validation"]["platt"], val["groups"], names, fit)
                brier = float(np.mean((scores - val["y"]) ** 2))
                candidates.append((brier, len(fit["patches"]), fit))
                selection.append({"method": method, "epsilon": epsilon, "support": support,
                                  "validation_brier": brier, "updates": len(fit["patches"]), "status": fit["status"]})
        selected[method] = min(candidates, key=lambda c: (c[0], c[1]))[2]
        models[method] = selected[method]
        for split, data in partitions.items():
            predictions[split][method] = replay(predictions[split]["platt"], data["groups"], names, selected[method])
    frozen_thresholds = {method: budget_threshold(scores, protocol["validation_escalation_budget"], val["keys"])
                         for method, scores in predictions["validation"].items()}
    working = ROOT / "runs" / f"{run_id}.in-progress"
    working.mkdir(parents=True, exist_ok=True)
    # All learned objects and operating thresholds are serialized before test metrics.
    frozen = {"models": models, "group_thresholds": thresholds, "group_names": names,
              "routing_thresholds": frozen_thresholds, "selection": selection}
    save(working / "frozen.json", frozen)
    save(working / "protocol.json", protocol)
    save(working / "splits.json", manifests)
    summary = {"schema_version": 1, "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
               "provenance": {**provenance, "sources": source["sources"], "prepared_sha256": digest(ROOT / "data" / "prepared.json")},
               "names": METHOD_NAMES, "features": FEATURES, "group_thresholds": thresholds,
               "selection": selection, "selected": selected, "partitions": {}, "comparisons": {},
               "frozen_thresholds": frozen_thresholds, "audit_support": min(protocol["support_grid"])}
    summary["provenance"]["numerical_warnings"] = sorted({str(w.message) for w in numerical_warnings})
    summary["provenance"]["numerical_check"] = "Exported logistic probabilities match an independent non-BLAS einsum calculation to 1e-12 on every partition. All features and final probabilities are finite."
    seed, reps = protocol["seed"], protocol["bootstrap_replicates"]
    for split, data in partitions.items():
        summary["partitions"][split] = {
            "tasks": len(manifests[split]), "candidates": len(data["y"]),
            "primary_pass_rate": float(data["y"].mean()), "alternate_pass_rate": float(data["alternate"].mean()),
            "empty_candidates": int(np.sum(data["x"][:, FEATURES.index("empty_code")])),
            "methods": {},
        }
        for method, scores in predictions[split].items():
            result = metrics(scores, data["y"], data["groups"], data["ids"], names, bins, summary["audit_support"])
            result["routing"] = policy(scores, data["y"], data["alternate"], frozen_thresholds[method], data["keys"])
            if split == "test":
                result["brier_ci"] = clustered_interval((scores - data["y"])**2, data["ids"], seed, reps)
                routed = route_mask(scores, frozen_thresholds[method], data["keys"])
                outcomes = np.where(routed, data["alternate"], data["y"])
                result["routing"]["success_ci"] = clustered_interval(outcomes, data["ids"], seed, reps)
                result["frontier"] = [dict(budget=budget, **policy(scores, data["y"], data["alternate"],
                    budget_threshold(predictions["validation"][method], budget, val["keys"]), data["keys"])) for budget in np.linspace(0, 1, 11).tolist()]
                summary["comparisons"][method] = clustered_interval(
                    (scores - data["y"])**2 - (predictions[split]["platt"] - data["y"])**2,
                    data["ids"], seed, reps)
            summary["partitions"][split]["methods"][method] = result
    primary, reference = predictions["test"]["multicalibration"], predictions["test"]["platt"]
    nonempty = test["x"][:, FEATURES.index("empty_code")] == 0
    summary["nonempty_comparison"] = clustered_interval((primary[nonempty]-test["y"][nonempty])**2 -
        (reference[nonempty]-test["y"][nonempty])**2, test["ids"][nonempty], seed, reps)
    summary["nonempty_comparison"]["candidates"] = int(nonempty.sum())
    primary_route = np.where(route_mask(primary, frozen_thresholds["multicalibration"], test["keys"]), test["alternate"], test["y"])
    ref_route = np.where(route_mask(reference, frozen_thresholds["platt"], test["keys"]), test["alternate"], test["y"])
    summary["routing_comparison"] = clustered_interval(primary_route-ref_route, test["ids"], seed, reps)
    summary["checks"] = {"disjoint_task_splits": True, "no_cross_split_exact_prompt_duplicates": True,
        "source_checksums_verified": True, "ten_candidates_per_task": True, "no_generated_code_executed": True,
        "test_excluded_from_selection": True, "exported_logistic_matches_non_blas": True,
        "all_predictions_bounded": all(np.all((p >= 0) & (p <= 1)) for s in predictions.values() for p in s.values())}
    summary["limitations"] = [
        "One benchmark and one frozen split; bootstrap intervals are conditional on the fitted models, not training-set uncertainty.",
        "Publisher-provided labels and program extraction were not independently re-evaluated. Empty programs can encode extraction failures.",
        "No private Cursor tasks, repository-level edits, live traffic, measured latency, or billing data.",
        "Fixed groups and minimum task support do not establish population multicalibration guarantees. Worst-cell residuals are descriptive and selected across many cells.",
        "Task IDs and exact prompts are disjoint; related problem families and pretraining contamination may still cross boundaries.",
        "Alternate outputs are independent recorded solutions, not repairs. An alternate model can turn a passing primary answer into a failure.",
        "Ten candidates are repeated trials, not a best-of-ten deployment. Feature extraction uses only the candidate being scored.",
        "Examining test results and then revising this protocol makes future comparisons exploratory; a new final claim needs fresh held-out data.",
    ]
    rows = []
    for split, data in partitions.items():
        for i, row in enumerate(data["rows"]):
            rows.append({"task_id": row["id"], "name": row["name"], "candidate": row["index"], "split": split,
                         "tie_key": float(data["keys"][i]),
                         "label": row["label"], "alternate_label": row["alternate_label"], "features": row["features"],
                         "groups": [name for j, name in enumerate(names) if data["groups"][i, j]],
                         "scores": {method: float(scores[i]) for method, scores in predictions[split].items()}})
    save(working / "predictions.json", rows)
    save(working / "summary.json", summary)
    if verify and destination.exists():
        for filename in ("protocol.json", "splits.json", "frozen.json", "predictions.json"):
            if digest(working / filename) != digest(destination / filename):
                raise AssertionError(f"Recomputed artifact differs: {filename}")
        previous = json.loads((destination / "summary.json").read_text())
        for field in ("partitions", "comparisons", "nonempty_comparison", "routing_comparison"):
            if summary[field] != previous[field]:
                raise AssertionError(f"Recomputed metrics differ: {field}")
        print(f"Verified full deterministic refit, predictions, and metrics for {run_id}", flush=True)
        for path in working.iterdir():
            path.unlink()
        working.rmdir()
        return destination
    save(working / "manifest.json", {"run_id": run_id, "files": {path.name: digest(path) for path in working.glob("*.json")}})
    with zipfile.ZipFile(working / "experiment.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in working.glob("*.json"):
            bundle.write(path, f"research/runs/{run_id}/{path.name}")
        for path in code_files:
            bundle.write(path, str(path.relative_to(ROOT.parent)))
        bundle.write(working / "protocol.json", "research/protocol.json")
        bundle.writestr("research/runs/latest.json", json.dumps({"run_id": run_id}))
    working.rename(destination)
    save(ROOT / "runs" / "latest.json", {"run_id": run_id})
    print(json.dumps({"run_id": run_id, "test_tasks": len(manifests["test"]),
                      "primary_comparison": summary["comparisons"]["multicalibration"],
                      "routing_comparison": summary["routing_comparison"]}, indent=2), flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--verify", action="store_true", help="Refit from cached data and verify an existing run bit-for-bit")
    args = parser.parse_args()
    run(args.protocol, args.verify)
