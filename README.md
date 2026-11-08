# Multicalibration for Coding Confidence

Tests whether an independent check improves learned subgroup corrections beyond simply using smaller updates.

## Quick Start

The [notebook](confirmation_analysis.ipynb) contains saved outputs and reads the included results offline. Use Python 3.13 for fresh feature extraction; offline checks also run on Python 3.11.

```bash
git clone https://github.com/Smokeybear10/coding-confidence-calibration.git
cd coding-confidence-calibration
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab confirmation_analysis.ipynb
```

## Design

The earlier experiment changed several things together when it added confirmation. This study freezes one tree-selected subgroup and separates two choices: where the correction is estimated, and whether an independent sample must confirm the proposal. The four matched arms share their rule, estimation task count, support check and step size.

Three controls select an update step, including zero, using the same validation tasks. One of them pools all correction tasks. Equal available label budgets are reported alongside per-policy query requirements. This isolates one correction rather than claiming a full iterative multicalibration procedure.

| Evidence | Size | Purpose |
| --- | --- | --- |
| Recorded LiveCodeBench outputs[^1] | 1,055 tasks; five outer folds | Exploratory source comparison |
| Known-probability simulations | 24 settings; 100 fitting seeds each | Separate selected noise, genuine bias and task dependence |
| MultiPL-E Python outputs[^1][^2] | 161 tasks; ten answers each | Evaluate frozen source-trained pipelines |

The [protocol](research/confirmation_protocol.json) was versioned before these evaluation runs. [Methods and related work](CONFIRMATION_STUDY.md) explain the matched estimates, data roles and controls.

## Findings

**Independent estimation matters before confirmation.** At the largest synthetic budget with task-shared label noise and calibrated starting scores, probability MSE is 0.001064 using a reused estimate, 0.000118 using an independent estimate, and 0.000002 after adding the gate.

**The gate can discard useful corrections.** With strong common bias and task-shared noise, gating increases MSE by 0.000565 relative to the same independent estimate without a gate; the paired interval is [0.000355, 0.000806]. With independent label draws at that budget, it accepts every proposal and changes nothing. Pooled, validation-tuned updates perform better in both strong-bias cases.

**External improvement is unestablished.** Platt scores without a subgroup update have the lowest observed external Brier. The gate-versus-ungated independent-estimate interval spans zero. Avoiding harmful corrections in a simulation does not establish a coding-benchmark gain.

| Method | Source OOF Brier | External ensemble Brier |
| --- | ---: | ---: |
| Raw logistic | 0.1071 | 0.0775 |
| Platt / no subgroup update | 0.1100 | 0.0740 |
| Reused estimate / fixed step | 0.1090 | 0.0761 |
| Independent estimate / fixed step | 0.1095 | 0.0748 |
| Reused estimate / gate | 0.1098 | 0.0745 |
| Independent estimate / gate | 0.1098 | 0.0745 |
| Reused estimate / tuned step | 0.1104 | 0.0765 |
| Independent estimate / tuned step | 0.1093 | 0.0745 |
| Pooled auditor / tuned step | 0.1105 | 0.0750 |

Lower is better. Source numbers average held-out fold scores at the 339-task audit/tuning budget; external numbers evaluate the five-pipeline probability ensemble on another task set. These are different evaluation summaries. [Source contrasts](confirmation_results/source_contrasts.csv) and [external contrasts](confirmation_results/external_contrasts.csv) retain the uncertainty estimates.

![Null correction rates and the incremental confirmation effect under task-correlated outcomes](confirmation_results/gate_diagnostics.png)

The [full synthetic grid](confirmation_results/simulation_summary.csv) includes weak and rare bias, independent labels, and all budgets. Improvements in the weak/rare task-shared cases mostly reflect retaining the original scores, rather than recovering the hidden bias.

## Reproduce

```bash
python -m research.external --output work/new-external --protocol research/confirmation_protocol.json
python -m research.confirmation_study --output work/new-confirmation --external work/new-external
python -m research.confirmation_figures --results work/new-confirmation
python -m unittest discover -s research -t . -v
```

Use fresh directories. Source and synthetic experiments run without `--external`; `--smoke` uses separate pilot seeds and skips external evaluation. The external adapter downloads about 46 MB of pinned public files. External records stay local; the repository publishes aggregate transfer results.

## Scope and Earlier Experiments

Fresh splits and conservative updates already exist in multicalibration implementations.[^3] This study tests their empirical tradeoffs under its fixed procedure and generators. Bootstrap confirmation is approximate. Synthetic intervals describe independent fitting-seed replications within each setting; source and external intervals are conditional/descriptive. There is no simultaneous-coverage claim. The source sample was already inspected; external benchmark contamination and semantic overlap remain unverified.

[Earlier calibration and routing experiments](advanced_analysis.ipynb) have their own [methods](ADVANCED_METHODS.md). The [original analysis](analysis.ipynb), [feature definitions](METHODS.md), [reproduction record](REPRODUCIBILITY.md) and [raw-data walkthrough](reconstruction/cursor-confidence-walkthrough.ipynb) preserve the initial experiment. All studies use public recorded outputs, separately from private Cursor work.

[^1]: [CALIBRI](https://huggingface.co/datasets/lavis-nlp/CALIBRI), revision `7a4a7dc7aceb1b4a426c48761d0f128f2daf6a54`.
[^2]: [MultiPL-E](https://github.com/nuprl/MultiPL-E). Its pinned upstream conditions are recorded by the evaluation-only adapter.
[^3]: [MCBoost implementation guidance](https://mlr-org.github.io/mcboost/articles/mcboost_basics_extensions.html); further primary references are in [CONFIRMATION_STUDY.md](CONFIRMATION_STUDY.md).

Built by Thomas Ou
