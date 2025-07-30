# Methods

## Observation, target and data

One observation is one task paired with one recorded primary coding answer. Its target is the publisher's `is_correct`, converted to 0 or 1. Passing is an outcome on the benchmark tests, not a guarantee of correctness on every input.

The pinned CALIBRI configurations are `livecodebench_qwen3` for primary answers and `livecodebench_gpt-oss` for alternates. All ten primary attempts per task are retained, rather than selecting the best attempt using hidden outcomes. Primary candidate j is paired with alternate candidate j for routing evaluation; this index pairing is arbitrary and is not a repair workflow.

The model sees neither task ID, difficulty annotations, alternate outcomes, nor other candidates' pass/fail labels. Task IDs enforce grouping and generate deterministic routing tie keys.

| Role | Tasks | Primary answers | Learned quantities |
| --- | ---: | ---: | --- |
| Training | 316 | 3,160 | Scaling, base coefficients, group quartiles |
| Calibration | 211 | 2,110 | Platt parameters, isotonic mapping, offsets |
| Validation | 264 | 2,640 | Configuration choice and routing cutoffs |
| Test | 264 | 2,640 | None |

Official validation and test sets remain intact. Training tasks are hash-sorted using seed 73129; the first 60% fit the base predictor, and the rest fit calibration. Every task's ten related attempts stay together. Exact normalized prompt duplicates are checked when preparing the source data.

## Predictors

The extractor parses generated Python as data. It never executes it. Count transforms use the natural logarithm `log(1 + count)`; standardization is learned separately on training data.

| Feature | Definition |
| --- | --- |
| `log_prompt_chars` | Log-transformed prompt characters |
| `log_code_chars` | Log-transformed extracted-code characters |
| `log_code_lines` | Log-transformed extracted-code lines |
| `log_output_tokens` | Log-transformed number of recorded token scores |
| `mean_logprob` | Average sampled-token log probability |
| `p10_logprob` | 10th percentile of sampled-token log probabilities |
| `logprob_sd` | Population standard deviation of those log probabilities |
| `low_probability_fraction` | Fraction with log probability below -2 |
| `ast_depth` | Maximum entire syntax-tree depth, root at zero |
| `branches` | Counts of if, conditional-expression and match nodes |
| `loops` | Counts of loops and comprehension generators |
| `functions` | Counts of regular and async function definitions |
| `imports` | Counts of import statements |
| `syntax_valid` | Nonempty extracted code successfully parses |
| `empty_code` | Extracted code is empty or whitespace |
| `missing_logprobs` | No sampled-token scores are available |

Generation statistics summarize the full response, which can include explanation, rather than only code tokens. These are sampled-token probabilities, so they do not determine full next-token entropy. Missing token records receive zero-valued log-probability summaries plus an explicit missingness flag. Empty or invalid programs remain in the sample.

Neither token entropy nor multi-sample self-consistency is implemented. Code complexity is a possible correlate of difficulty, not evidence that making a program shorter would cause it to pass.

## Base model and score-only calibration

StandardScaler estimates training means and scales. Binary logistic regression uses L2 regularization with C=1, `max_iter=2000`, and `random_state=0`. Its sigmoid output estimates benchmark success. Coefficients are descriptive model parameters, without a causal or significance interpretation.

Platt scaling keeps the base model fixed and fits:

$$p_{\mathrm{Platt}}=\sigma\left(a\,\operatorname{logit}(\operatorname{clip}(p_0,10^{-6},1-10^{-6}))+b\right).$$

Its one-input logistic fit uses C=1e6 and calibration labels. The recorded slope is positive. This transformation preserves ordering apart from clipping ties; changing the numeric scale does not necessarily change a rank-based routing policy.

Isotonic regression fits a nondecreasing mapping from base score to calibration outcomes, clipping out-of-range inputs to its endpoint predictions.

## Groups and conditional corrections

There are seven allowed groups:

| Group | Observable membership |
| --- | --- |
| `all` | Every answer |
| `long_prompt` | Prompt-length feature exceeds its training 75th percentile |
| `long_code` | Code-length feature exceeds its training 75th percentile |
| `nested_code` | AST depth exceeds its training 75th percentile |
| `uses_imports` | At least one import statement |
| `empty_code` | Empty extracted program |
| `long_prompt&nested_code` | Both conditions |

Five equal-width bands partition the current probability, with 1 included in the last band. For group G and band b, define:

$$S_{G,b}=\{i:i\in G,\;p_i\in b\},\qquad r_{G,b}=\frac{1}{|S_{G,b}|}\sum_{i\in S_{G,b}}(y_i-p_i).$$

At each iteration, scan supported slices and select the largest absolute mean residual. Update matching probabilities:

$$p_i\leftarrow\operatorname{clip}(p_i+r_{G,b},0,1).$$

Save the group, band and offset in order, recompute current score bands, and scan again. Support is the number of distinct tasks represented in a slice, not the count of repeated attempts. The allowed tolerance grid is {0.03, 0.06, 0.10}; minimum-task support is {20, 40}; the cap is 100 updates. Validation Brier selects among six fits, preferring fewer patches on an exact tie.

The recorded choice has tolerance 0.03, support 40, and 12 updates. This is a finite, predefined group search. It does not fit a regression tree or discover unrestricted subgroup definitions. Inference replays frozen rules against each evolving score, without labels or additional fitting.

Empirical stopping on eligible calibration cells does not establish the same bound on new tasks. Groups overlap and supported cells can change with score movement. The reported worst-slice diagnostic uses a common audit support of 20 tasks; it is not identical to the selected fit criterion.

## Comparisons

| Method | Purpose |
| --- | --- |
| Logistic | Full sixteen-feature base predictor |
| Platt | Shared score-only adjustment |
| Isotonic | Flexible monotone score-only adjustment |
| Multicalibration | Group-and-current-band offsets after Platt |
| Score bins | Same correction procedure restricted to `all` |
| Group-aware logistic | Original features plus six non-`all` group indicators |
| Token only | Output-token count, four token-score summaries, missing-score flag |
| Structure only | Complement of the token-only feature columns |
| Token likelihood | Exponentiated average token log probability, a heuristic |

The group-aware baseline tests whether putting group information into the original predictor is competitive. The ablations remove entire feature families; they are not individual-feature causal analyses. No forest, XGBoost model, or learned subgroup auditor is evaluated here.

## Evaluation and routing

Brier is the mean of $(p-y)^2$. Log loss uses probabilities clipped to [1e-6, 1-1e-6]. ECE weights absolute residuals within the five global score bands by candidate count. Worst supported residual takes the maximum over eligible group/band intersections; it is sensitive to bin boundaries, support, and searching many cells.

Paired differences subtract Platt's per-answer loss before bootstrapping. Each of 1,000 bootstrap samples resamples 264 whole test tasks, preserving their ten dependent attempts. These conditional intervals do not include all retraining or model-selection uncertainty, and no simultaneous-coverage claim is made across comparisons.

The default routing cutoff targets the lowest-confidence 30% of validation answers. The cutoff contains a score and a deterministic tie key. Test coverage is allowed to differ. The routing curve repeats this procedure at eleven fixed validation budgets, then evaluates the resulting frozen cutoffs on test data. It does not select cutoffs from test outcomes.

Accepted error means failures divided by accepted primary answers. Cascade pass rate also counts the recorded alternate's outcome after escalation. A primary success replaced by an alternate failure is counted as a regression. No cost penalty, latency measurement, or optimal economic routing claim is part of the implemented policy.

## Interpretation

The question is whether post-processing adds useful probability or decision quality beyond a reasonable predictor. The recorded comparisons do not establish a clear Brier gain from Platt or subgroup correction. The same default routing outcomes also prevent attributing an operational improvement to either calibration stage in this run.

Empty extracted programs, publisher-label limitations, benchmark contamination, reused test results and numeric bin-boundary sensitivity constrain the interpretation. The [reproduction record](REPRODUCIBILITY.md) reports those numerical checks. This implementation is an application of existing methods and does not claim the original multicalibration paper's population guarantees.
