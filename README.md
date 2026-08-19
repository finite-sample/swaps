# Validation-Swap Distributions

This repository contains a methodological note and exact finite-population analysis for auditing machine-scored outcomes in experiments.

Researchers may score every experimental outcome with a machine-learning model or large language model while obtaining human labels for only a probability sample. Overall prediction accuracy does not reveal whether scoring errors distort the treatment contrast. This project studies the distribution of treatment-effect estimates induced by which experimental clusters receive gold-standard labels.

For a linear treatment contrast, the average fixed-budget raw-swap path is mechanically determined by the proxy and gold endpoints. Its dispersion and tails reveal whether the correction is diffuse, concentrated in a few clusters, or offsetting across clusters.

## Statistical objects

For cluster contribution $\psi_g$, proxy estimate $\widehat\tau_P$, and a simple random validation set $S_k$ of $k$ clusters, the raw-swap estimate is

$$
\widehat\tau_R(S_k)
=
\widehat\tau_P+\sum_{g\in S_k}\psi_g.
$$

Its mean is forced to interpolate between the proxy and gold estimates:

$$
E\lbrace\widehat\tau_R(S_k)\rbrace
=
\widehat\tau_P+
\frac{k}{G}(\widehat\tau_G-\widehat\tau_P).
$$

The design-scaled estimate

$$
\widehat\tau_H(S_k)
=
\widehat\tau_P+
\frac{G}{k}\sum_{g\in S_k}\psi_g
$$

is a standard Horvitz-Thompson difference correction and a simple prediction-powered estimator. The paper does not claim a new correction estimator. Its contribution is to use the full fixed-budget distribution as an estimand-specific validation diagnostic.

Four formal propositions sharpen the diagnostic. A sharp Cauchy-Schwarz bound, $|\widehat\tau_G-\widehat\tau_P|\le r\sqrt{N(1/N_1+1/N_0)}$, shows accuracy metrics constrain the endpoint gap only at the order of the RMSE $r$: the bound is 0.40 in the generated design against a gold effect of 0.10. The $k=1$ swap distribution identifies the multiset of cluster contributions, which determines every fixed-budget distribution, while the mean path depends only on their sum. The sample variance of the observed contributions is design-unbiased for $S_\psi^2$, so one validation draw prices the variance of every counterfactual budget. And by the Hajek negligibility condition, the swap distribution is approximately normal only when no small set of clusters dominates the centered squared contributions, so moment-based budget planning fails exactly in the concentrated case the diagnostic exists to detect.

## Numerical design

The generated example contains 20 equal-size experimental clusters and exactly enumerates every fixed-size validation set. Three machine-scoring error patterns share the same gold outcomes, predictive RMSE, and predictive $R^2$:

- Diffuse errors allocate the gold-minus-proxy treatment-effect gap equally across clusters.
- Concentrated errors allocate the same gap to two clusters.
- Offsetting errors have large positive and negative cluster contributions but a zero endpoint gap.

The first two cases have the same proxy effect, gold effect, predictive accuracy, and mean swap path. Their random-swap distributions differ sharply.

## Repository layout

```text
replicate_final.py        Exact enumeration, tables, and LaTeX macros
validation_swaps.tex      Paper source
references.bib            Bibliography, validated against Crossref and arXiv
validation_swaps.pdf      Compiled paper
generated/                Reproducible CSVs and LaTeX fragments
tests/                    Econometric identities and reproducibility checks
pyproject.toml            Runtime and development dependencies
Makefile                  Analysis, checks, paper build, and Docker CI
```

## Reproduce the analysis

Python 3.11 or newer is required.

```bash
uv sync --extra dev
uv run make analysis
```

The analysis writes:

- `table_scenarios.csv` and `table_scenarios.tex`
- `table_budgets.csv` and `table_budgets.tex`
- `macros.tex`
- `metadata.json`

The manuscript inputs all tables and repeated numerical results from these generated files.

For a smaller functional run:

```bash
uv run make analysis-fast
```

Command-line overrides are available for the cluster count, cluster size, seed, and output directory:

```bash
uv run python replicate_final.py \
  --clusters 20 \
  --cluster-size 50 \
  --seed 2024 \
  --outdir generated
```

Exact enumeration accepts at most 22 clusters. Larger designs require a sampling approximation, which this release does not implement.

## Validate the repository

```bash
uv run make check
uv run make paper
```

`make check` runs Black, isort, Flake8, Ruff, and pytest. `make paper` regenerates the exact analysis and compiles the manuscript twice. The local container target uses the standard Python 3.13 image:

```bash
make ci-docker
```

## Scope

The exact swap distribution is available when gold-minus-proxy discrepancies are observed for the finite validation pool, as in a benchmark dataset or a pilot used to study smaller future budgets. With one small validation sample, unseen cluster contributions cannot be recovered nonparametrically. Population inference requires known validation inclusion probabilities or a stated transport model. Swap quantiles are not confidence intervals for the experimental treatment effect.
