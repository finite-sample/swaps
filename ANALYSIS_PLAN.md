# Covariate correction paths: analysis plan

## Question and scope

Does a finite covariate-correction path reveal sensitivity of a downstream
regression coefficient that prediction RMSE and additive influence approximations
miss? The object is the OLS coefficient on treatment in a fixed dataset. A player
is a row or a declared batch of rows whose proxy covariates can be replaced with
verified values. The endpoint corrects only those players, not unverified rows.

This is an exploratory methodological benchmark, not a preregistered study.
Previous work in this repository and the October 17 research audit have already
shown finite-correction interactions and the original implementation's
Shapley/Banzhaf mismatch. The specifications below are recorded before running
the new benchmark. No empirical superiority or novel mathematical identity is
assumed.

## Implementation and comparison

Enumerate every correction subset for small games. Report the exact distribution
at each fixed number of corrected players and the Bernoulli path obtained by
mixing those distributions. Calculate Shapley values with factorial coalition
weights, Banzhaf values separately, and marginal effects conditional on coalition
size. Use a permutation estimator with Monte Carlo standard errors for larger
games; every permutation must telescope to the endpoint difference.

Compare full refits with (a) exact single-player changes summed additively,
(b) paired-score changes retaining the quadratic covariate correction term but
holding the inverse Gram matrix fixed, and (c) the derivative in feature error
amplitude. The paired-score approximation is the principal influence baseline.
All methods target the same finite coefficient and use the same correction sets.
Report rank deficiency and conditioning explicitly; never silently substitute a
pseudoinverse or discard difficult coalitions.

## Synthetic benchmark

Hold the outcome, true confounder, treatment and additional covariate fixed across
three proxy-error patterns: diffuse, concentrated in a correction batch, and
aligned with treatment. Normalize each error vector to the same RMSE. Generate
treatment conditional on the true confounder and control, then an additive linear
outcome with an independent disturbance and a treatment coefficient of 0.5.
The full-data model identifies that coefficient by construction, but the
diagnostic compares realized regression coefficients, not population causal bias.

The main display uses seed 2024, 96 rows, eight correction batches and proxy RMSE
0.75. Batches keep enumeration feasible; they are not treatment-assignment
clusters and do not imply that validation must occur in clusters. The core API
also supports individual rows and incomplete validation pools.

The repeated benchmark uses seeds 0 through 19, all three error patterns, and
RMSE levels 0.05, 0.25, 0.75 and 1.5. Report every design and any failures. Summarize
coalition-level approximation RMSE, largest discrepancy, path curvature relative
to endpoint interpolation, and conditioning. These are descriptive benchmark
quantities. Bands across coalitions are correction-design quantiles, not
confidence intervals; repeated datasets describe this specified simulation only.

## Interpretation and falsification

Zero corrections must give a constant path. Outcome-only corrections with fixed
regressors must be additive. Exact Shapley values must sum to the endpoint gap.
Integrated marginal paths must agree with Shapley values. Small perturbations
should make appropriate first-order approximations accurate. Compare numerical
normal-equation updates with independent least-squares refits.

If the paired-score approximation tracks the refits well, report that result.
If discrepancies appear only under poor conditioning or extreme measurement
error, state that scope. A new sampling or domain-selection advantage is not
established by this benchmark. Learned risk scores, certified domains, regression
calibration intervals and causal-inference guarantees require separate designs;
the original unaudited pipeline will not be presented as validated evidence.

## Outputs and review

Generate CSV summaries, figures, manuscript tables and numerical macros from one
analysis pipeline. Preserve the existing linear-outcome benchmark as a boundary
case. Obtain an independent technical review, run local formatting and tests,
compile the paper, and inspect every rendered page. Record any changes to these
specifications and their reasons below.

## Deviations

Before running: independent review clarified that the concentrated scenario is
an additive control with one active batch. It remains in the design unchanged.
Attributions are specific to the declared batches and need not equal sums of
row-level attributions. Conditioning refers to the design matrix, not its Gram
matrix. Both influence approximations generally have second-order error in small
feature-error amplitude; no higher-order accuracy is claimed for paired scores.

## Post-run validation, October 17, 2026

The full run completed all 240 repeated designs without a rank failure. The
independent technical review checked the finite normal-equation identity,
comparison baselines, attribution weights, partial-pool endpoints, and the
manuscript's scope. Its numerical-variance finding was repaired by centering
coalition values before computing path moments, with an offset-invariance test.
Successful repeated-design results are saved even if another design fails.

The local checks cover 31 tests and Black, isort, Flake8 and Ruff. Full and fast
analysis runs completed. The paper was compiled with BibTeX and every page was
rendered for inspection; plot labels were enlarged after the first inspection.

The audit's data-integrity checks concern simulated datasets, matched error
magnitudes, complete coalition enumeration, and persistence of all declared
replications. Survey weights, linkage errors and observational-data provenance
are inapplicable to this synthetic benchmark. Causal identification holds only
by the specified simulation construction. No population confidence intervals,
trained-model performance, pilot transport, learned domain, or prospective
sampling-policy benefit was evaluated. The historical calibration and domain
audit findings remain unresolved for the original implementation, which is
preserved separately and is not used to support the new paper.

## Paired-archive pilot, October 17, 2026

This exploratory extension uses two archives selected for paired predictions and
reference measurements, before fitting their downstream regressions. At this
point their source code, schemas, row counts, and housing missingness have been
inspected; no downstream coefficients or approximation errors have been viewed.
This is a self-attested specification, not a preregistration or a held-out test
of a method selected on these data. Report both archives, including null findings.

Housing: use all 46,418 complete paired rows in Kluger's formatted MOSAIKS data.
Fit HousingPrice on an intercept, Income, Nightlights, and RoadLength. Target the
Income coefficient; correct both Nightlights and RoadLength together for each
selected row. Outcome and Income must match exactly between paired datasets.
Keep the archive's transformations. Report the coefficient per one proxy-sample
standard deviation of Income (population SD convention, denominator n). Center
and scale all other regressors using fixed proxy-sample means and SDs for numerical
stability; these transforms remain fixed throughout the correction path.

Reddit: fit score, clamped to [-10, 10] as in Fong and Tyler, on an intercept and
incivility in the 613-row validation file. Train the archive's weighted radial
SVM on its separate 2,412-row training set and fixed 2,172-column document-term
matrices. Use its specified seed and defaults without tuning on validation data.
Check that comment IDs are unique and the two splits do not overlap. Predictors
are text features only; the score is never used to train the classifier. Correct
predicted incivility with the human reference label. Target the binary incivility
coefficient in score points. This diagnostic uses the validation sample alone;
it does not reproduce the paper's much larger primary-sample analysis.

Both targets are descriptive associations. Housing illustrates correcting
adjustment covariates; Reddit illustrates correcting the regressor of interest.
Neither archive identifies the effect of repairing confounders on a causal
estimate. Reference measurements are treated as fixed for this exercise, not
asserted to be error-free constructs. Reject missing or nonfinite required inputs
rather than silently changing samples.

Players are individual rows, including rows whose proxy already equals its
reference. At budgets 1%, 5%, 10%, 25%, 50%, 75%, and 100% of rows (rounded to the
nearest integer), compare direct least-squares refits with the three established
additive approximations. Use 200 independently drawn random permutations, seed
20261017, and their prefixes so that each budget is uniformly sampled without
replacement; compute the common full-correction endpoint only once. Randomness
describes which observed rows are corrected, not statistical uncertainty about
a population. Do not select rows using their observed correction effects.

Primary outputs: signed endpoint shifts and approximation errors in coefficient
units; mean, SD, 5th/95th percentiles of refitted coefficients by budget; and
approximation RMSE and maximum absolute error on the same sampled sets. Also
report errors relative to the endpoint shift when nonzero, but never use that
potentially small denominator alone. As a descriptive scale, report the full
reference-fit HC0 standard error; it is not a valid population uncertainty claim
without a sampling/dependence argument. No significance tests or causal coverage
claims are planned. Thresholds of 0.1 and 0.5 of that scale summarize numerical
size only, not scientific importance. No direction or size of discrepancy is
assumed. Accurate additive approximations are evidence against the practical
need for joint refits in that setting.

Compute exact individual-row changes from updated OLS sufficient statistics,
checking a fixed random sample of 100 of them against independent least-squares
refits. Evaluate all multirow correction sets by direct least squares, and record
conditioning and failures. Check the no-error boundary and row/column alignment.
Freeze raw input checksums and source versions; preserve R and Python versions,
classifier settings, training overlap checks, and derived-data hashes. Keep raw
text and row identifiers out of generated public result tables. This pilot does
not evaluate adaptive sampling, required validation size, or ASI superiority.

Before fitting, independent review added the following checks: retrieve the
original CSV bytes and preserve their archived order when pairing labels with
document-term rows (the archive provides no independent row-ID map for those
matrices); retain the original sparse binary-presence construction; supplement
100 random singleton checks with the 20 highest-leverage and 20 largest-update
rows; report Monte Carlo standard errors for means and RMSE; and label maxima as
sampled maxima. Exact-singleton discrepancy is R(S); influence discrepancies also
include linearization error. Accurate approximations on these random sets do not
exclude rare or deliberately targeted interactions. Classifier settings are
explicit in prepare_archives.R and installed package versions are saved.

Data-integrity correction before fitting: the original training CSV contains
2,413 rows; Dataverse's converted tabular download contains 2,412. The original
document-term indices extend through 2,412 (zero-based), matching the original
CSV. Use all 2,413 original training rows. The validation CSV and converted table
both contain the same 613 IDs in the same order. The earlier 2,412-row count was
based on the converted table and is superseded. This changes no validation target.

Post-run: both archive analyses completed all 1,201 correction sets (200 draws at
six interior budgets and one endpoint), without rank failures. The independent
review reproduced the original classifier defaults and found zero disagreements
with the 613 prepared validation predictions. The largest singleton-update
discrepancy from direct refits was below 2e-15 in both archives. The main paper
includes these as exploratory illustrations and reports accurate approximations
at small random budgets alongside discrepancies at larger budgets. No additional
outcomes, subgroups, budgets, or tuned classifiers were searched after results.
