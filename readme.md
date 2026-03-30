# Partial Credit: Diagnosing Proxy Covariates with Validation Swaps

When a regression uses a proxy $\hat{X}$ instead of the true covariate $X$, how much does it matter — and where? This paper introduces a validation-swap framework that answers both questions using an internal validation sample where both $X$ and $\hat{X}$ are observed.

## The idea

Swap validated truths into the proxy design matrix, row by row, and watch the coefficient move.

- **Swap path**: traces $\hat{\beta}_1$ as a function of the fraction of validated rows swapped in. Flat means the proxy is fine. Steep means it isn't.
- **SIM (Swap Importance Measure)**: a Shapley-value decomposition that identifies *which rows* drive the proxy-induced distortion.
- **Portable risk score**: predicts SIM from observables so the analyst can define a proxy-safe domain on the full sample, not just the validation subset.
- **Audit gate**: a held-out validation check that accepts or rejects the proposed domain. If it rejects, fall back to the validation-only estimate.

The framework is diagnostic first. It tells you whether the proxy matters, where it matters, and whether borrowing proxy-only observations is worth the bias cost. It does not claim distribution-free inference or universal consistency.

## Key finding

At typical validation sample sizes ($n_{\text{val}}$ in the hundreds), the validation-only estimate is already precise enough that adding proxy data with a distortion budget $\delta$ makes the confidence interval *wider*, not tighter. The precision case for domain borrowing is narrow: it arises when each validated observation is genuinely expensive ($n_{\text{val}} < 170$ at $\delta = 0.05$). The framework's main value is diagnostic.

## Repository contents

```
├── validation_swaps.tex        # Paper (LaTeX source)
├── validation_swaps.pdf        # Paper (compiled)
├── replicate_final.py          # Replication script (all tables and figures)
├── fig1_swap_paths.png         # Figure 1: swap paths across scenarios
├── fig2_sim_anatomy.png        # Figure 2: SIM vs leverage × proxy error
└── README.md
```

## Replication

Requires Python 3.8+ with `numpy`, `scipy`, and `matplotlib`. No other dependencies.

```bash
# Quick check (~90 seconds)
python replicate_final.py --fast

# Full replication (~4 minutes, reproduces all paper numbers)
python replicate_final.py

# Custom settings
python replicate_final.py --n_sims 500 --outdir my_results
```

Outputs are written to `results/` (or the directory specified by `--outdir`):

| File | Contents |
|------|----------|
| `fig1_swap_paths.png` | Swap paths across three scenarios |
| `fig2_sim_anatomy.png` | SIM vs leverage × proxy error |
| `fig3_bias_bars.png` | Bias by method and scenario |
| `table_7_2_main.csv` | Table 1: main results |
| `table_7_4_precision.csv` | Precision comparison (confirms the arithmetic in Section 7.2) |
| `table_7_5a_audit_threshold.csv` | Table 2: audit threshold sensitivity |
| `table_7_5b_val_size.csv` | Table 3: validation size sensitivity |

## Three scenarios

The simulations use three DGPs designed to illustrate different failure modes:

**Benign.** Small homogeneous measurement error. The proxy works everywhere. The pipeline confirms this and lets the analyst proceed.

**Heterogeneous.** Proxy quality varies with an observable covariate ($|W_2|$). A correctly specified global correction dominates. The pipeline's value here is diagnostic: the swap path and SIM show *why* the proxy fails in the tails of $W_2$, which a global correction hides.

**Ugly.** The proxy captures only part of $X$ and misses an outcome-relevant latent component. Global correction makes things worse ($-80\%$ bias). Both SIM variants correctly refuse to accept any domain. The pipeline falls back to the validation-only estimate — the right answer.

## Citation

```
@unpublished{partialcredit2025,
  title={Partial Credit: Diagnosing Proxy Covariates with Validation Swaps},
  year={2025}
}
```