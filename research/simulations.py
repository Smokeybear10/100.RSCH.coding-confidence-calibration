from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .auditors import apply_auditor, fit_auditor
from .data import ROOT, digest


def synthetic_tasks(tasks: int, seed: int, biased: bool, candidates: int = 10) -> dict:
    rng = np.random.default_rng(seed)
    task_features = rng.normal(size=(tasks, 16))
    x = np.repeat(task_features, candidates, axis=0)+rng.normal(scale=.25, size=(tasks*candidates, 16))
    p = expit(.7*x[:, 0])
    region = (x[:, 1] > 0) & (x[:, 2] > 0)
    truth = np.clip(p-.25*region if biased else p, 0, 1)
    shared_uniform = np.repeat(rng.uniform(size=tasks), candidates)
    labels = (shared_uniform < truth).astype(int)
    return {"x": x, "p": p, "truth": truth, "y": labels,
            "ids": np.repeat([f"{seed}:{task}" for task in range(tasks)], candidates)}


def simulate(output: Path, protocol_path: Path = ROOT / "advanced_protocol.json", smoke: bool = False) -> dict:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    code_hashes = {f"research/{path.name}": digest(path) for path in sorted(ROOT.glob("*.py"))}
    config = protocol["simulation"]
    settings = protocol["auditor"]
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    rows = []
    maximum = max(config["calibration_tasks"])
    for seed in seeds:
        for scenario, biased in (("calibrated_null", False), ("hidden_interaction", True)):
            calibration = synthetic_tasks(maximum, seed, biased, config["candidates_per_task"])
            test = synthetic_tasks(config["test_tasks"], seed+100000, biased, config["candidates_per_task"])
            identifiers = np.unique(calibration["ids"])
            discovery_ids = identifiers[:int(maximum*2/3)]
            confirmation_ids = identifiers[int(maximum*2/3):]
            for total in config["calibration_tasks"]:
                first, second = int(total*2/3), total-int(total*2/3)
                discovery = {key: value[np.isin(calibration["ids"], discovery_ids[:first])] for key, value in calibration.items()}
                confirmation = {key: value[np.isin(calibration["ids"], confirmation_ids[:second])] for key, value in calibration.items()}
                pool = {key: np.concatenate([discovery[key], confirmation[key]]) for key in discovery}
                fitted = {"tree_mc": fit_auditor(pool, None, settings, seed),
                          "confirmed_tree": fit_auditor(discovery, confirmation, settings, seed)}
                scores = {"raw": test["p"], **{name: apply_auditor(test["x"], test["p"], fit) for name, fit in fitted.items()}}
                for method, p in scores.items():
                    rows.append({"seed": seed, "scenario": scenario, "calibration_tasks": total,
                                 "method": method, "known_probability_mse": float(np.mean((p-test["truth"])**2)),
                                 "realized_brier": float(np.mean((p-test["y"])**2)),
                                 "patches": len(fitted[method]["patches"]) if method != "raw" else 0,
                                 "any_patch": bool(fitted[method]["patches"]) if method != "raw" else False})
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "simulation_runs.csv", index=False)
    summary = frame.groupby(["scenario", "calibration_tasks", "method"], as_index=False).agg(
        known_probability_mse=("known_probability_mse", "mean"),
        mse_seed_sd=("known_probability_mse", "std"),
        realized_brier=("realized_brier", "mean"), mean_patches=("patches", "mean"),
        any_patch_rate=("any_patch", "mean"), seeds=("seed", "nunique"),
    )
    summary.to_csv(output / "simulation_summary.csv", index=False)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "smoke": smoke,
              "protocol_sha256": protocol_hash, "rows": len(frame), "code_sha256": code_hashes,
              "scope": "Synthetic known-probability negative control and interaction example, not coding-benchmark evidence",
              "dependence": "A shared uniform draw couples each task's ten binary outcomes while preserving each marginal probability",
              "estimator": "Mean squared difference from the generator's known conditional probability on independently generated test tasks",
              "limitations": "Finite Monte Carlo test sample and ten simulation seeds; no universal error-control guarantee"}
    (output / "simulation.json").write_text(json.dumps(report, indent=2)+"\n")
    (output / "protocol.json").write_bytes(protocol_bytes)
    manifest = {path.name: digest(path) for path in output.iterdir() if path.is_file()}
    (output / "manifest.json").write_text(json.dumps({"files": manifest}, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Known-truth calibration controls with correlated task outcomes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "advanced_protocol.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(simulate(args.output, args.protocol, args.smoke), indent=2))
