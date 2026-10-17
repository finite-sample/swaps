# Covariate Correction Paths

**Which prediction errors change the econometric result?** This repository replaces selected proxy covariates with verified values, refits the regression, and records how the target coefficient changes. It studies the whole distribution across correction sets, not just prediction accuracy or the fully corrected endpoint.

The current benchmark concerns a treatment coefficient adjusted for a mismeasured confounder. The statistical tools are established: the random-subset mean path is a diagonal of Owen's multilinear extension, and Shapley values allocate its endpoint change. The research question is whether applying these tools to finite covariate corrections reveals sensitivity missed by simpler diagnostics. This repository does not establish a new optimal validation design or a population inference guarantee.

## What it computes

- Exact regression refits for every subset of a small, declared validation pool.
- Mean and dispersion when each correction is included independently with probability `p`, and distributions at a fixed correction budget.
- Exact Shapley attributions, separately labeled Banzhaf values, and permutation Shapley estimates with Monte Carlo standard errors.
- Comparisons against additive single-repair changes, a finite paired-score approximation, and a derivative in feature-error amplitude.

For covariates, replacing measurements changes both the regression score and its inverse Gram matrix. That can make corrections interact. For a linear outcome contrast with fixed weights, corrections are additive; the older outcome benchmark remains as this boundary case.

## Results and limits

Three synthetic proxies have identical RMSE but different correction paths. The treatment-aligned example has substantial curvature and approximation error without near-singular regression designs. The one-active-batch control is exactly additive: its mean path is straight, although its correction-set dispersion is large. The stronger paired-score formula does **not** uniformly outperform the feature derivative. All results and the repeated 240-design benchmark are generated from code.

Two paired replication archives add empirical illustrations: housing with predicted adjustment covariates, and Reddit with predicted incivility as the regressor of interest. Exact isolated corrections and paired scores track small random correction sets closely. Their discrepancies grow as corrections accumulate. Both applications concern descriptive associations; neither establishes a better verification policy or a causal estimation gain. The archive pilot uses individual rows as correction units.

These are conditional diagnostics of a known validation pool. Unknown gold values cannot be recovered from the path, and correction-set quantiles are not confidence intervals. A partial pool's endpoint corrects that pool only. Batch-level attributions depend on the grouping and generally are not sums of row-level Shapley values. The simulations prescribe proxy errors; they do not train an ML model or evaluate a prospective acquisition policy.

## Reproduce

Python 3.11 or newer, R with Matrix, SparseM and e1071, `curl`, and a LaTeX installation with BibTeX are needed for the complete paper build. The Python checks and synthetic benchmarks do not require R. Install missing R packages with `Rscript -e 'install.packages(c("Matrix", "SparseM", "e1071"), repos="https://cloud.r-project.org")'`.

```bash
uv sync --extra dev
uv run make check
uv run make analysis
uv run make paper
```

`make check` runs Black, isort, Flake8, Ruff, and pytest. `make analysis-fast` runs smaller functional benchmarks in `/tmp`; `make ci-docker` runs checks and those benchmarks with the standard Python 3.13 image. The paper build regenerates the synthetic and archive analyses and compiles the bibliography.

Run the covariate benchmark alone:

```bash
uv run python replicate_covariates.py
uv run python replicate_covariates.py --fast --outdir /tmp/covariate-swaps-fast
```

Run the archive pilot alone:

```bash
uv run make archives
```

This downloads checksum-pinned public inputs, reproduces the archived Reddit classifier on its separate training split, and evaluates 200 random correction paths in each dataset. Downloads and derived row-level data are cached under ignored `data/archives/`; aggregate results and sampled-set diagnostics go to `generated/archives/`. Source URLs, versions, and SHA-256 hashes are in `research_notes/archives/sources.json`. The Reddit preparation deliberately uses the original CSV: Dataverse's converted training table drops a row and does not align with the archived document-term matrix. Conditioning, Monte Carlo error, and all observed approximation errors are retained. Correction-set quantiles are not population confidence intervals, and the reference-fit HC0 scale is only a descriptive normalization.

## Use the diagnostic

```python
from covariate_swaps import CorrectionGame, enumerate_game

# Designs include the intercept and every regressor. The arrays have the same
# shape; only rows in the declared players need verified measurements.
game = CorrectionGame(
    proxy=proxy_design,
    verified=verified_design,
    outcome=y,
    players=((0,), (1,), (2,)),
    target=1,
)
exact = enumerate_game(game)
mean, sd = exact.bernoulli([0.0, 0.5, 1.0])
shapley = exact.shapley()
fixed_budget_values = exact.fixed_size()[2]
```

Exact enumeration is limited to 16 players because computation grows exponentially. A player can be a row or a disjoint batch. `permutation_shapley(game, permutations=1000, seed=0)` approximates attributions for larger games; it does not approximate the complete correction-set distribution. A rank-deficient correction design raises an error rather than silently changing the estimand.

## Files

| File | Purpose |
| --- | --- |
| `covariate_swaps.py` | Correction games, refits, paths, attribution and influence baselines |
| `replicate_covariates.py` | Matched-RMSE and repeated simulation benchmark |
| `replicate_archives.py`, `prepare_archives.R` | Pinned archive retrieval, original classifier, and row-level correction diagnostics |
| `replicate_final.py` | Retained linear-outcome boundary case |
| `ANALYSIS_PLAN.md` | Specifications recorded before the new benchmark |
| `validation_swaps.tex`, `validation_swaps.pdf` | Manuscript and compiled paper |
| `generated/covariates/` | Coalition values, summaries, figures and manuscript inputs |
| `generated/archives/` | Paired-archive diagnostics, figures, table, and provenance |
| `generated/` | Retained outcome benchmark |
| `reports/`, `research_notes/` | Literature review and historical implementation audit |
| `tests/` | Mathematical identities, numerical refits and build checks |

The original March 2026 implementation is preserved under `research_notes/Original validation swaps novelty/original_snapshot/`. Its attribution and audit issues are documented in the accompanying report; the current implementation is separately tested.
