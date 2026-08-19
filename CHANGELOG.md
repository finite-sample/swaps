# Changelog

## Unreleased

- Add four formal propositions to the paper: a sharp Cauchy-Schwarz bound on the endpoint gap at fixed RMSE, multiset sufficiency and mean-path coarseness, design-unbiased estimation of the budget-variance path from one validation draw, and the Hajek negligibility condition for Normal approximation of the swap distribution.
- Add `worst_case_gap` and `hajek_ratio` with enumeration-backed tests for each proposition, and new manuscript macros for the bound and the concentration shares.
- Replace the inline bibliography with a Crossref-validated `references.bib`, switch to natbib author-year citations, and add the missing foundational literature (Horvitz-Thompson, Neyman two-phase sampling, the survey difference estimator, PPI++, active statistical inference, and LLM annotation).
- Cut the figures entirely: the constructed example's evidence is exact numbers, so the budget table now carries them, gaining a Normal-price column that shows the variance-based price erring in both directions while the exact price is non-monotone in the budget. matplotlib is no longer a dependency.
- Demote the mean-path/multiset proposition to a prose remark; reframe the abstract around pricing validation labeling designs.
- Rebuild the front of the paper as a base-case-then-boundary argument: under individual randomization, non-differential scoring error is balanced by design and only treatment-dependent error biases the contrast; a new section states the three things clustered assignment breaks, with the arm-imbalance identity test-gated.
- Position the diagnostic against dropping-data robustness metrics, and report the sign-reversal probability the manuscript had promised but never shown.
- Fix README math to GitHub-rendered `$`/`$$` delimiters.

## v0.1.0 - 2026-08-18

- Introduce exact validation-swap distributions for machine-scored experimental outcomes.
- Derive raw and design-scaled swap estimators and distinguish them from wild cluster bootstrap inference.
- Add exact finite-population examples, generated figures and tables, and the compiled methodological note.
- Add econometric identity tests, reproducibility checks, local automation, and GitHub Actions CI.
- Bound exact enumeration to 22 clusters to prevent accidental memory exhaustion.
