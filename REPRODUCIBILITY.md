# Reproduction check: 2026-11-06

This publication check used Python 3.13.5 on macOS arm64 with the pinned `requirements.txt`. It was performed with AI assistance. The original reference remains under `research/runs/a8630092fa9e/`; new outputs are written separately.

The later repeated-fold extension is documented in [ADVANCED_METHODS.md](ADVANCED_METHODS.md). Its numerical artifacts and manifests live separately in `advanced_results/` and `simulation_results/`. The agreement table below concerns the original reference only, not that extension.

## What is included

The archive contains 10,550 candidate records: sixteen input measurements, primary and alternate labels, task/candidate identifiers, partition assignments, tie keys, group memberships, and the nine methods' original probabilities. Raw prompts, token streams, generated programs and benchmark tests are not included in this numerical archive.

`predictions.json.gz` compresses the original prediction file. Verification hashes its decoded bytes against the original manifest, along with the frozen model, protocol, split manifest and summary. Reference JSON files match their recorded hashes, including their original lack of a trailing newline. The initial notebook-only publication had added harmless terminal newlines; the archive-completion commit restored byte-level compatibility.

The reference is a saved public-data study, not recovered proof of an earlier fellowship implementation. The earlier raw-data walkthrough was authored October 14, 2026; the main offline analysis notebook and these checks were added November 6, 2026. Both document an already-inspected sample.

## Three reproducibility levels

| Level | Inputs | What it checks |
| --- | --- | --- |
| Frozen replay | Included feature rows and saved models | Label-free probability mapping, metrics and routing |
| Feature refit | Included feature rows, recorded labels and splits | Base fitting, calibration, configuration choice and cutoffs |
| Source rebuild | Six pinned public Parquet files | Source parsing and feature extraction, followed by fitting |

Publication verified the first two levels. It did not freshly download all six source files or re-extract every record. Generated programs were never executed; correctness labels remain the publisher's results.

```bash
python -m research.reproduce --output work/replay
python -m research.reproduce --refit --output work/refit
python -m research.figures --results work/refit --output work/refit/diagnostics.png
```

Use unused output paths. The commands refuse to overwrite a nonempty directory or write into the saved reference run. Each output contains the protocol, tables, an environment/code-hash record, probability comparisons, and metric differences. A refit also saves its fitted objects.

## Measured agreement

| Comparison with reference | Frozen replay | Feature refit |
| --- | ---: | ---: |
| Maximum absolute probability difference, all methods and partitions | 5.5622e-14 | 4.1546e-11 |
| Maximum Brier difference | 1.1102e-16 | 3.3945e-14 |
| Maximum log-loss difference | 3.8858e-16 | 7.8881e-14 |
| Maximum five-bin ECE difference | 5.6872e-4 | 2.3457e-14 |
| Maximum worst-slice difference | 1.1102e-16 | 1.8774e-13 |

All probability comparisons passed an absolute tolerance of 1e-8. This is numerical agreement within tolerance, not a bitwise identity claim. The [replay record](reproduction/2026-11-06-replay.json) and [refit record](reproduction/2026-11-06-refit.json) retain actual differences, source hashes and environment versions. [Metric comparisons](results/metric_comparison.csv) retain the per-method/per-partition detail.

## Why one binned metric changes

For calibration-set isotonic predictions, replay moves two scores from exactly 0.4 and 0.6 to approximately 0.3999999999999966 and 0.5999999999999822. Those changes cross exact band boundaries. The recorded calibration ECE is approximately 4.18e-17; the replayed value is 0.0005687203791469, or 0.056872 percentage points.

The changes are not material probability differences, but ECE is discontinuous at the bin boundaries. Recomputing diagnostics from the original stored scores reproduces the original values. Replay tables keep the independently computed probabilities and expose the changed diagnostic; no rounding is introduced to force agreement. The held-out headline values remain unchanged at the precision reported in the README.

## Dependencies and run identities

Install `requirements.txt` for the current notebook and offline checks. `requirements-research.txt` is the historical dependency record, and is retained because the original provenance hashes reference it. In particular, the recorded run lists joblib 1.6.0 and threadpoolctl 3.7.0, whereas the publication environment uses 1.4.2 and 3.5.0. The current environment's exact versions are retained in each reproduction JSON.

`research.run` builds its identity from source files, protocol and dependency versions. A current source rebuild can therefore receive a different run ID even when the mathematical outputs agree. New raw rebuilds can be directed outside the reference tree:

```bash
python -m research.run --output work/raw-rebuild
```

This downloads approximately 466 MB, verifies pinned source hashes, and caches extraction under the ignored `research/data/`. Each content-addressed run writes its frozen objects before test metrics. The `--verify` option checks an existing run produced by that same current source/environment identity; it does not select the historical reference ID automatically.

## Checks

```bash
python -m unittest discover -s research -t . -v
jupyter nbconvert --to notebook --execute analysis.ipynb --output checked-analysis.ipynb --output-dir work
```

Tests verify archive checksums, disjoint task splits, support counted by distinct tasks, clipping and ordered replay, deterministic routing ties, validation-only configuration selection, archived metric calculations, and prediction invariance when test outcomes are changed. Missing reference records fail the suite rather than silently skipping the integration checks.

Replaying and refitting a reused benchmark does not create an independent experiment. Bootstrap uncertainty remains conditional on the fitted models and recorded task population. A new performance claim would require a frozen procedure and new evaluation data.
