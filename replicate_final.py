"""
Validation Swaps: Replication Script
=====================================

Reproduces all tables and figures from the paper.

Tables:
  Table 1   Main results (3 scenarios x 4 methods)
  Table 2   Audit threshold sensitivity
  Table 3   Validation size sensitivity

Figures:
  Figure 1  Swap paths (3 scenarios)
  Figure 2  SIM anatomy (3 scenarios)
  Figure 3  Bias bar chart

Usage:
    python replicate_final.py              # default (~4 min)
    python replicate_final.py --fast       # quick check (~90s)

Requires: numpy, scipy, matplotlib (no other dependencies)
"""

import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ═════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════

DEFAULT = dict(
    n=2000,
    n_val=500,
    n_sims_main=100,       # Table 7.2
    n_sims_sens=60,        # Tables 7.4, 7.5
    B_swap=100,            # swap-path MC draws
    B_sim=80,              # SIM MC draws
    delta=0.15,
    seed=2024,
)

FAST = dict(
    n=2000,
    n_val=500,
    n_sims_main=50,
    n_sims_sens=30,
    B_swap=60,
    B_sim=60,
    delta=0.15,
    seed=2024,
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Replicate all tables/figures from Validation Swaps")
    ap.add_argument("--fast", action="store_true",
                    help="Reduced MC for quick check")
    ap.add_argument("--outdir", type=str, default="results",
                    help="Output directory")
    return ap.parse_args()


# ═════════════════════════════════════════════════════════
# Data-generating process
# ═════════════════════════════════════════════════════════

def dgp(n, n_val, scenario, rng):
    """
    Generate (X, Xhat, W, Y, val_idx).

    Scenarios
    ---------
    'benign':        Xhat = X + N(0, 0.3^2)
    'heterogeneous': Xhat = X + sigma(W2)*nu, sigma = 0.1 + 1.5|W2|
    'ugly':          Y includes latent U; Xhat = W2 + 0.5*nu
    """
    W2 = rng.standard_normal(n)
    W3 = rng.standard_normal(n)
    W = np.column_stack([W2, W3])

    if scenario == "benign":
        X = rng.standard_normal(n)
        Xhat = X + rng.standard_normal(n) * 0.3
        Y = 1.0 * X + 0.5 * W2 + 0.3 * W3 + rng.standard_normal(n)

    elif scenario == "heterogeneous":
        X = rng.standard_normal(n)
        sigma_e = 0.1 + 1.5 * np.abs(W2)
        Xhat = X + rng.standard_normal(n) * sigma_e
        Y = 1.0 * X + 0.5 * W2 + 0.3 * W3 + rng.standard_normal(n)

    elif scenario == "ugly":
        U = rng.standard_normal(n)
        X = W2 + U
        Xhat = W2 + rng.standard_normal(n) * 0.5
        Y = (1.0 * X + 0.5 * W2 + 0.3 * W3
             + 1.0 * U + rng.standard_normal(n))

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    val_idx = rng.choice(n, n_val, replace=False)
    return X, Xhat, W, Y, val_idx


# ═════════════════════════════════════════════════════════
# OLS with HC1 standard errors
# ═════════════════════════════════════════════════════════

def ols(x_col, W, Y, subset=None):
    """
    OLS of Y on (1, x_col, W).
    Returns (beta_1, se_1) where beta_1 is the coefficient on x_col.
    """
    if subset is not None:
        x_col = x_col[subset]
        W = W[subset]
        Y = Y[subset]
    n = len(Y)
    M = np.column_stack([np.ones(n), x_col, W])
    p = M.shape[1]
    try:
        Hinv = np.linalg.inv(M.T @ M)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    beta = Hinv @ (M.T @ Y)
    e = Y - M @ beta
    meat = (M.T * e ** 2) @ M * (n / (n - p))
    V = Hinv @ meat @ Hinv
    return beta[1], np.sqrt(max(V[1, 1], 0.0))


# ═════════════════════════════════════════════════════════
# Hybrid design operator
# ═════════════════════════════════════════════════════════

def hybrid(X, Xhat, mask):
    """Rows where mask=True use X, rest use Xhat."""
    out = Xhat.copy()
    out[mask] = X[mask]
    return out


# ═════════════════════════════════════════════════════════
# Swap path
# ═════════════════════════════════════════════════════════

def swap_path(X, Xhat, W, Y, val_idx, p_grid, B=80, rng=None):
    """
    Estimate phi(p) = E[beta_1(X^{(S)})] for each p.
    Returns (means, sds).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(X)
    means = []
    sds = []
    for p in p_grid:
        if p == 0.0:
            b, _ = ols(Xhat, W, Y)
            means.append(b)
            sds.append(0.0)
        elif p == 1.0:
            mask = np.zeros(n, dtype=bool)
            mask[val_idx] = True
            b, _ = ols(hybrid(X, Xhat, mask), W, Y)
            means.append(b)
            sds.append(0.0)
        else:
            bs = []
            for _ in range(B):
                mask = np.zeros(n, dtype=bool)
                mask[val_idx[rng.random(len(val_idx)) < p]] = True
                b, _ = ols(hybrid(X, Xhat, mask), W, Y)
                if not np.isnan(b):
                    bs.append(b)
            means.append(np.mean(bs) if bs else np.nan)
            sds.append(np.std(bs) if bs else np.nan)
    return np.array(means), np.array(sds)


# ═════════════════════════════════════════════════════════
# SIM (Shapley-value swap importance)
# ═════════════════════════════════════════════════════════

def compute_sim(X, Xhat, W, Y, val_idx, B=120, rng=None):
    """
    Monte Carlo Shapley approximation of SIM_i for each i in val_idx.
    Returns array of length len(val_idx).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(X)
    nv = len(val_idx)
    sim = np.zeros(nv)
    counts = np.zeros(nv)
    k = min(20, nv)

    for _ in range(B):
        mask = np.zeros(n, dtype=bool)
        mask[val_idx[rng.random(nv) < 0.5]] = True
        b0, _ = ols(hybrid(X, Xhat, mask), W, Y)
        if np.isnan(b0):
            continue
        for ci in rng.choice(nv, size=k, replace=False):
            i = val_idx[ci]
            mask2 = mask.copy()
            mask2[i] = not mask2[i]
            b1, _ = ols(hybrid(X, Xhat, mask2), W, Y)
            if np.isnan(b1):
                continue
            sim[ci] += (b0 - b1) if mask[i] else (b1 - b0)
            counts[ci] += 1

    counts[counts == 0] = 1
    return sim / counts


# ═════════════════════════════════════════════════════════
# SIM prediction (portable risk score)
# ═════════════════════════════════════════════════════════

def predict_sim(abs_sim_train, Xhat_train, W_train, Xhat_all, W_all):
    """
    Fit r(Z) ≈ E[|SIM| | Z] via OLS with quadratic features.
    Predict for all rows.
    """
    def feats(xh, w):
        return np.column_stack([
            np.ones(len(xh)),
            np.abs(xh),
            w,
            w[:, 0] ** 2,
            w[:, 1] ** 2,
            w[:, 0] * w[:, 1],
        ])

    try:
        g = np.linalg.lstsq(
            feats(Xhat_train, W_train), abs_sim_train, rcond=None
        )[0]
        return np.maximum(feats(Xhat_all, W_all) @ g, 0.0)
    except Exception:
        return np.full(len(Xhat_all), np.mean(abs_sim_train))


# ═════════════════════════════════════════════════════════
# Domain selection
# ═════════════════════════════════════════════════════════

def select_domain_val_only(abs_sim, disc_idx, X, Xhat, W, Y, delta):
    """
    Domain = validated rows with |SIM| <= tau.
    Returns (indices, tau) or (None, nan).
    """
    quantiles = np.linspace(0.3, 1.0, 15)
    best = None
    best_n = 0
    best_tau = np.nan
    for q in quantiles:
        tau = np.quantile(abs_sim, q)
        in_dom = disc_idx[abs_sim <= tau]
        if len(in_dom) < 20:
            continue
        bx, _ = ols(X, W, Y, in_dom)
        bxh, _ = ols(Xhat, W, Y, in_dom)
        if np.isnan(bx) or np.isnan(bxh):
            continue
        if abs(bx - bxh) <= delta and len(in_dom) > best_n:
            best = in_dom
            best_n = len(in_dom)
            best_tau = tau
    return best, best_tau


def select_domain_projected(abs_sim, disc_idx, X, Xhat, W, Y, delta, n):
    """
    Domain = all rows with predicted |SIM| <= tau.
    Returns (boolean mask, tau) or (zeros, nan).
    """
    pred = predict_sim(abs_sim, Xhat[disc_idx], W[disc_idx], Xhat, W)
    quantiles = np.linspace(0.3, 1.0, 15)
    best_mask = None
    best_n = 0
    best_tau = np.nan
    for q in quantiles:
        tau = np.quantile(pred, q)
        candidate = pred <= tau
        vi = disc_idx[candidate[disc_idx]]
        if len(vi) < 20:
            continue
        bx, _ = ols(X, W, Y, vi)
        bxh, _ = ols(Xhat, W, Y, vi)
        if np.isnan(bx) or np.isnan(bxh):
            continue
        if abs(bx - bxh) <= delta and np.sum(candidate) > best_n:
            best_mask = candidate
            best_n = np.sum(candidate)
            best_tau = tau
    if best_mask is None:
        return np.zeros(n, dtype=bool), np.nan
    return best_mask, best_tau


# ═════════════════════════════════════════════════════════
# Local correction
# ═════════════════════════════════════════════════════════

def local_correct(X_val, Xhat_val, W_val, Xhat_target, W_target,
                  val_in_domain):
    """
    Fit mu(Z) = E[X | Z, R=1] on validation rows in domain.
    Predict for target rows.
    """
    if np.sum(val_in_domain) < 10:
        return Xhat_target.copy()
    M = np.column_stack([
        np.ones(np.sum(val_in_domain)),
        Xhat_val[val_in_domain],
        W_val[val_in_domain],
    ])
    try:
        g = np.linalg.lstsq(M, X_val[val_in_domain], rcond=None)[0]
    except Exception:
        return Xhat_target.copy()
    M_pred = np.column_stack([
        np.ones(len(Xhat_target)),
        Xhat_target,
        W_target,
    ])
    return M_pred @ g


# ═════════════════════════════════════════════════════════
# SE(D) via paired influence functions
# ═════════════════════════════════════════════════════════

def distortion_with_se(X, Xhat, W, Y, subset):
    """
    D = beta_1(X) - beta_1(Xhat) on subset, with SE(D).
    SE computed via influence-function difference, retaining covariance.
    """
    x = X[subset]
    xh = Xhat[subset]
    w = W[subset]
    y = Y[subset]
    n = len(y)
    p = w.shape[1] + 2

    Mx = np.column_stack([np.ones(n), x, w])
    Mxh = np.column_stack([np.ones(n), xh, w])
    try:
        Hx = np.linalg.inv(Mx.T @ Mx)
        Hxh = np.linalg.inv(Mxh.T @ Mxh)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    bx = Hx @ Mx.T @ y
    bxh = Hxh @ Mxh.T @ y
    D = bx[1] - bxh[1]

    ex = y - Mx @ bx
    exh = y - Mxh @ bxh
    psi = (Hx @ (Mx.T * ex))[1, :] - (Hxh @ (Mxh.T * exh))[1, :]
    se_D = np.sqrt(max(np.sum(psi ** 2) * n / (n - p), 0.0))
    return D, se_D


# ═════════════════════════════════════════════════════════
# One MC replication (produces one row per method)
# ═════════════════════════════════════════════════════════

def run_one(scenario, cfg, rng):
    """
    Full pipeline for one MC draw.
    Returns dict with all quantities needed for tables.
    """
    n = cfg["n"]
    n_val = cfg["n_val"]
    delta = cfg["delta"]
    B_sim = cfg["B_sim"]

    X, Xhat, W, Y, val_idx = dgp(n, n_val, scenario, rng)

    # Split validation: discovery / audit
    rng.shuffle(val_idx)
    half = n_val // 2
    disc_idx = val_idx[:half]
    audit_idx = val_idx[half:]

    # ── Baselines ──
    b_oracle, se_oracle = ols(X, W, Y)
    b_naive, se_naive = ols(Xhat, W, Y)
    b_val, se_val = ols(X, W, Y, val_idx)

    # Global correction: E[X|Z] = g0 + g1*Xhat + g2*W2 + g3*W3
    Mfit = np.column_stack([np.ones(n_val), Xhat[val_idx], W[val_idx]])
    try:
        gam = np.linalg.lstsq(Mfit, X[val_idx], rcond=None)[0]
        mu_global = np.column_stack([np.ones(n), Xhat, W]) @ gam
        b_gcorr, se_gcorr = ols(mu_global, W, Y)
    except Exception:
        b_gcorr, se_gcorr = np.nan, np.nan

    # ── SIM on discovery fold ──
    sim_scores = compute_sim(X, Xhat, W, Y, disc_idx, B=B_sim, rng=rng)
    abs_sim = np.abs(sim_scores)

    res = dict(
        b_oracle=b_oracle, se_oracle=se_oracle,
        b_naive=b_naive, se_naive=se_naive,
        b_val=b_val, se_val=se_val,
        b_gcorr=b_gcorr, se_gcorr=se_gcorr,
    )

    # ── Approach A: val-only domain ──
    dom_v, tau_v = select_domain_val_only(
        abs_sim, disc_idx, X, Xhat, W, Y, delta
    )
    res["valonly_pass"] = False
    res["valonly_share"] = 0.0
    res["valonly_bias"] = np.nan
    res["valonly_b_or_dom"] = np.nan
    res["valonly_se_dom"] = np.nan

    if dom_v is not None:
        # Audit: recompute SIM on audit fold, apply same tau
        sim_aud = compute_sim(
            X, Xhat, W, Y, audit_idx,
            B=max(B_sim // 3, 30), rng=rng
        )
        audit_in = audit_idx[np.abs(sim_aud) <= tau_v]
        if len(audit_in) >= 15:
            D_a, _ = distortion_with_se(X, Xhat, W, Y, audit_in)
            if not np.isnan(D_a) and abs(D_a) <= delta:
                res["valonly_pass"] = True
                all_dom = np.concatenate([dom_v, audit_in])
                res["valonly_share"] = len(all_dom) / n
                b_or_dom, _ = ols(X, W, Y, all_dom)
                # Local correction
                vmask = np.isin(val_idx, set(all_dom))
                mu = local_correct(
                    X[val_idx], Xhat[val_idx], W[val_idx],
                    Xhat[all_dom], W[all_dom], vmask
                )
                b_corr, se_corr = ols(mu, W[all_dom], Y[all_dom])
                res["valonly_bias"] = b_corr - b_or_dom
                res["valonly_b_or_dom"] = b_or_dom
                res["valonly_se_dom"] = se_corr

    # ── Approach B: projected domain ──
    dom_mask, tau_p = select_domain_projected(
        abs_sim, disc_idx, X, Xhat, W, Y, delta, n
    )
    res["proj_pass"] = False
    res["proj_share"] = 0.0
    res["proj_bias"] = np.nan
    res["proj_b_or_dom"] = np.nan
    res["proj_se_dom"] = np.nan

    if not np.isnan(tau_p):
        audit_in_p = audit_idx[dom_mask[audit_idx]]
        if len(audit_in_p) >= 15:
            D_a, se_D_a = distortion_with_se(X, Xhat, W, Y, audit_in_p)
            if not np.isnan(D_a) and abs(D_a) <= delta:
                res["proj_pass"] = True
                dom_rows = np.where(dom_mask)[0]
                res["proj_share"] = len(dom_rows) / n
                b_or_dom, _ = ols(X, W, Y, dom_rows)
                vmask = np.array([dom_mask[v] for v in val_idx])
                mu = local_correct(
                    X[val_idx], Xhat[val_idx], W[val_idx],
                    Xhat[dom_rows], W[dom_rows], vmask
                )
                b_corr, se_corr = ols(mu, W[dom_rows], Y[dom_rows])
                res["proj_bias"] = b_corr - b_or_dom
                res["proj_b_or_dom"] = b_or_dom
                res["proj_se_dom"] = se_corr

    return res


# ═════════════════════════════════════════════════════════
# Table 7.2: Main results
# ═════════════════════════════════════════════════════════

def table_main(cfg):
    """Reproduce Table 7.2."""
    print("\n" + "=" * 78)
    print("TABLE 7.2: MAIN RESULTS")
    print("=" * 78)
    scenarios = ["benign", "heterogeneous", "ugly"]
    all_res = {}

    for scen in scenarios:
        print(f"\n  Running {scen}...")
        results = []
        for sim in range(cfg["n_sims_main"]):
            if (sim + 1) % 25 == 0:
                print(f"    {sim + 1}/{cfg['n_sims_main']}")
            rng = np.random.default_rng(cfg["seed"] + sim * 997
                                        + hash(scen) % 9999)
            results.append(run_one(scen, cfg, rng))
        all_res[scen] = results

    # Print
    print("\n" + "-" * 78)
    print(f"{'Scenario':>15} {'Method':>28} {'Bias':>8} "
          f"{'%Bias':>8} {'Pass':>6} {'n_p':>5} {'Share':>7}")
    print("-" * 78)

    csv_rows = []
    for scen in scenarios:
        res = all_res[scen]
        b_or = np.nanmean([r["b_oracle"] for r in res])
        ns = len(res)

        for label, key in [
            ("Naive", "b_naive"),
            ("Global correction", "b_gcorr"),
            ("Validation only", "b_val"),
        ]:
            vals = [r[key] for r in res if not np.isnan(r[key])]
            bias = np.mean(vals) - b_or if vals else np.nan
            pct = 100 * bias / b_or if (vals and abs(b_or) > 1e-8) else np.nan
            print(f"{scen:>15} {label:>28} {bias:>8.3f} "
                  f"{pct:>8.1f} {'':>6} {'':>5} {'':>7}")
            csv_rows.append([scen, label, f"{bias:.4f}", f"{pct:.1f}",
                             "", "", ""])

        for lbl, pkey, bkey, skey, okey in [
            ("SIM (val only)+corr", "valonly_pass", "valonly_bias",
             "valonly_share", "valonly_b_or_dom"),
            ("SIM (projected)+corr", "proj_pass", "proj_bias",
             "proj_share", "proj_b_or_dom"),
        ]:
            passed = [r for r in res if r[pkey]]
            n_pass = len(passed)
            rate = n_pass / ns
            if passed:
                biases = [r[bkey] for r in passed if not np.isnan(r[bkey])]
                oracles = [r[okey] for r in passed if not np.isnan(r[okey])]
                shares = [r[skey] for r in passed]
                mb = np.mean(biases) if biases else np.nan
                pct = (100 * mb / np.mean(oracles)
                       if biases and oracles and abs(np.mean(oracles)) > 1e-8
                       else np.nan)
                ms = np.mean(shares)
                print(f"{scen:>15} {lbl:>28} {mb:>8.4f} "
                      f"{pct:>8.1f} {rate:>6.0%} {n_pass:>5} {ms:>7.2f}")
                csv_rows.append([scen, lbl, f"{mb:.4f}", f"{pct:.1f}",
                                 f"{rate:.3f}", str(n_pass), f"{ms:.2f}"])
            else:
                print(f"{scen:>15} {lbl:>28} {'--':>8} "
                      f"{'--':>8} {rate:>6.0%} {n_pass:>5} {'--':>7}")
                csv_rows.append([scen, lbl, "", "", f"{rate:.3f}",
                                 str(n_pass), ""])

    # CSV
    with open(os.path.join(cfg["outdir"], "table_7_2_main.csv"), "w") as f:
        f.write("scenario,method,bias,pct_bias,pass_rate,n_pass,share\n")
        for r in csv_rows:
            f.write(",".join(r) + "\n")

    return all_res


# ═════════════════════════════════════════════════════════
# Table 7.4: Precision comparison
# ═════════════════════════════════════════════════════════

def table_precision(cfg):
    """Reproduce Table 7.4."""
    print("\n" + "=" * 78)
    print("TABLE 7.4: PRECISION COMPARISON (heterogeneous)")
    print("=" * 78)
    deltas = [0.05, 0.10, 0.15, 0.20, 0.30]
    n_rep = cfg["n_sims_sens"]

    print(f"{'delta':>6} {'Pass%':>7} {'n_p':>5} {'SE_dom':>8} "
          f"{'SE_val':>8} {'CI_dom':>8} {'CI_val':>8} {'Tighter?':>9}")
    print("-" * 68)

    csv_rows = []
    for dg in deltas:
        cfg_d = {**cfg, "delta": dg}
        passed = []
        for sim in range(n_rep):
            rng = np.random.default_rng(cfg["seed"] + sim * 997 + 444)
            r = run_one("heterogeneous", cfg_d, rng)
            if r["proj_pass"]:
                passed.append(r)

        rate = len(passed) / n_rep
        if passed:
            se_d = np.mean([r["proj_se_dom"] for r in passed
                            if not np.isnan(r["proj_se_dom"])])
            se_v = np.mean([r["se_val"] for r in passed
                            if not np.isnan(r["se_val"])])
            ci_d = 1.96 * se_d + dg
            ci_v = 1.96 * se_v
            tighter = "Yes" if ci_d < ci_v else "No"
            print(f"{dg:>6.2f} {rate:>7.0%} {len(passed):>5} {se_d:>8.3f} "
                  f"{se_v:>8.3f} {ci_d:>8.3f} {ci_v:>8.3f} {tighter:>9}")
            csv_rows.append([dg, rate, len(passed), se_d, se_v,
                             ci_d, ci_v, tighter])
        else:
            print(f"{dg:>6.2f} {rate:>7.0%} {0:>5} {'--':>8} "
                  f"{'--':>8} {'--':>8} {'--':>8} {'--':>9}")
            csv_rows.append([dg, rate, 0, "", "", "", "", ""])

    with open(os.path.join(cfg["outdir"], "table_7_4_precision.csv"), "w") as f:
        f.write("delta,pass_rate,n_pass,se_domain,se_val,"
                "ci_domain,ci_val,domain_tighter\n")
        for r in csv_rows:
            f.write(",".join(str(x) for x in r) + "\n")


# ═════════════════════════════════════════════════════════
# Table 7.5a: Audit threshold sensitivity
# ═════════════════════════════════════════════════════════

def table_audit_threshold(cfg):
    """Reproduce Table 7.5a."""
    print("\n" + "=" * 78)
    print("TABLE 7.5a: AUDIT THRESHOLD (heterogeneous, projected)")
    print("=" * 78)
    deltas = [0.05, 0.10, 0.15, 0.25, 0.50]
    n_rep = cfg["n_sims_sens"]

    print(f"{'delta':>6} {'Pass%':>7} {'n_p':>5} "
          f"{'Bias|pass':>10} {'Share|pass':>11}")
    print("-" * 48)

    csv_rows = []
    for dg in deltas:
        cfg_d = {**cfg, "delta": dg}
        passed = []
        for sim in range(n_rep):
            rng = np.random.default_rng(cfg["seed"] + sim * 997 + 111)
            r = run_one("heterogeneous", cfg_d, rng)
            if r["proj_pass"]:
                passed.append(r)

        rate = len(passed) / n_rep
        if passed:
            biases = [r["proj_bias"] for r in passed
                      if not np.isnan(r["proj_bias"])]
            shares = [r["proj_share"] for r in passed]
            mb = np.mean(biases) if biases else np.nan
            ms = np.mean(shares)
            print(f"{dg:>6.2f} {rate:>7.0%} {len(passed):>5} "
                  f"{mb:>10.4f} {ms:>11.2f}")
            csv_rows.append([dg, rate, len(passed), mb, ms])
        else:
            print(f"{dg:>6.2f} {rate:>7.0%} {0:>5} "
                  f"{'--':>10} {'--':>11}")
            csv_rows.append([dg, rate, 0, "", ""])

    with open(os.path.join(cfg["outdir"],
                            "table_7_5a_audit_threshold.csv"), "w") as f:
        f.write("delta,pass_rate,n_pass,bias_if_pass,share_if_pass\n")
        for r in csv_rows:
            f.write(",".join(str(x) for x in r) + "\n")


# ═════════════════════════════════════════════════════════
# Table 7.5b: Validation size sensitivity
# ═════════════════════════════════════════════════════════

def table_val_size(cfg):
    """Reproduce Table 7.5b."""
    print("\n" + "=" * 78)
    print("TABLE 7.5b: VALIDATION SIZE (heterogeneous, projected, delta=0.15)")
    print("=" * 78)
    val_sizes = [200, 400, 600, 800]
    n_rep = cfg["n_sims_sens"]

    print(f"{'n_val':>6} {'Pass%':>7} {'n_p':>5} "
          f"{'Bias|pass':>10} {'Share|pass':>11}")
    print("-" * 48)

    csv_rows = []
    for nv in val_sizes:
        cfg_v = {**cfg, "n_val": nv}
        passed = []
        for sim in range(n_rep):
            rng = np.random.default_rng(cfg["seed"] + sim * 997 + 222)
            r = run_one("heterogeneous", cfg_v, rng)
            if r["proj_pass"]:
                passed.append(r)

        rate = len(passed) / n_rep
        if passed:
            biases = [r["proj_bias"] for r in passed
                      if not np.isnan(r["proj_bias"])]
            shares = [r["proj_share"] for r in passed]
            mb = np.mean(biases) if biases else np.nan
            ms = np.mean(shares)
            print(f"{nv:>6} {rate:>7.0%} {len(passed):>5} "
                  f"{mb:>10.4f} {ms:>11.2f}")
            csv_rows.append([nv, rate, len(passed), mb, ms])
        else:
            print(f"{nv:>6} {rate:>7.0%} {0:>5} "
                  f"{'--':>10} {'--':>11}")
            csv_rows.append([nv, rate, 0, "", ""])

    with open(os.path.join(cfg["outdir"],
                            "table_7_5b_val_size.csv"), "w") as f:
        f.write("n_val,pass_rate,n_pass,bias_if_pass,share_if_pass\n")
        for r in csv_rows:
            f.write(",".join(str(x) for x in r) + "\n")


# ═════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════

def figure_swap_paths(cfg):
    """Figure 1: swap paths across scenarios."""
    print("\n  Figure 1: swap paths")
    scenarios = ["benign", "heterogeneous", "ugly"]
    p_grid = np.linspace(0, 1, 9)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, scen in zip(axes, scenarios):
        rng = np.random.default_rng(cfg["seed"] + hash(scen) % 9999)
        X, Xhat, W, Y, vi = dgp(cfg["n"], cfg["n_val"], scen, rng)
        m, s = swap_path(X, Xhat, W, Y, vi, p_grid,
                         B=cfg["B_swap"], rng=rng)
        ax.plot(p_grid, m, "o-", color="#E53935", markersize=5)
        ax.fill_between(p_grid, m - s, m + s, alpha=0.15, color="#E53935")
        ax.axhline(1.0, color="gray", ls=":", alpha=0.4)
        ax.set_xlabel("swap fraction p")
        ax.set_title(scen.capitalize())
    axes[0].set_ylabel("$\\hat{\\beta}_1(p)$")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["outdir"], "fig1_swap_paths.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_sim_anatomy(cfg):
    """Figure 2: SIM anatomy across scenarios."""
    print("  Figure 2: SIM anatomy")
    scenarios = ["benign", "heterogeneous", "ugly"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, scen in zip(axes, scenarios):
        rng = np.random.default_rng(cfg["seed"] + hash(scen) % 9999)
        X, Xhat, W, Y, vi = dgp(cfg["n"], cfg["n_val"], scen, rng)
        sim = compute_sim(X, Xhat, W, Y, vi, B=cfg["B_sim"], rng=rng)
        err = np.abs(X[vi] - Xhat[vi])
        Mh = np.column_stack([np.ones(len(vi)), Xhat[vi], W[vi]])
        try:
            H = Mh @ np.linalg.inv(Mh.T @ Mh) @ Mh.T
            lev = np.diag(H)
        except np.linalg.LinAlgError:
            lev = np.ones(len(vi))
        interact = lev * err
        sc = ax.scatter(interact, np.abs(sim), alpha=0.35, s=10,
                        c=np.abs(W[vi, 0]), cmap="RdYlGn_r",
                        edgecolors="none", vmin=0, vmax=3)
        r = np.corrcoef(interact, np.abs(sim))[0, 1]
        ax.set_xlabel("leverage × |proxy error|")
        ax.set_title(f"{scen.capitalize()} (r={r:.2f})")
    axes[0].set_ylabel("|SIM|")
    fig.colorbar(sc, ax=axes[-1], label="|W₂|", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["outdir"], "fig2_sim_anatomy.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_bias_bars(all_res, cfg):
    """Figure 3: bias bar chart from main results."""
    print("  Figure 3: bias bars")
    scenarios = ["benign", "heterogeneous", "ugly"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, scen in zip(axes, scenarios):
        res = all_res[scen]
        b_or = np.nanmean([r["b_oracle"] for r in res])
        ns = len(res)
        methods = []
        biases = []
        colors = []

        b = np.nanmean([r["b_naive"] for r in res])
        methods.append("Naive")
        biases.append(b - b_or)
        colors.append("#F44336")

        vals = [r["b_gcorr"] for r in res if not np.isnan(r["b_gcorr"])]
        if vals:
            methods.append("Global\ncorr")
            biases.append(np.mean(vals) - b_or)
            colors.append("#FF9800")

        methods.append("Val\nonly")
        biases.append(np.nanmean([r["b_val"] for r in res]) - b_or)
        colors.append("#2196F3")

        for pkey, bkey, color, lbl in [
            ("valonly_pass", "valonly_bias", "#4CAF50", "SIM\n(val)"),
            ("proj_pass", "proj_bias", "#9C27B0", "SIM\n(proj)"),
        ]:
            passed = [r for r in res if r[pkey]]
            if passed:
                bs = [r[bkey] for r in passed if not np.isnan(r[bkey])]
                if bs:
                    rate = len(passed) / ns
                    methods.append(f"{lbl}\n({rate:.0%})")
                    biases.append(np.mean(bs))
                    colors.append(color)

        x = np.arange(len(methods))
        ax.bar(x, biases, color=colors, alpha=0.7, edgecolor="white")
        ax.axhline(0, color="gray", ls="-", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=7)
        ax.set_title(scen.capitalize())
    axes[0].set_ylabel("Bias")
    fig.suptitle(
        "Bias by method (domain methods conditional on audit pass)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["outdir"], "fig3_bias_bars.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════

def main():
    args = parse_args()
    cfg = FAST.copy() if args.fast else DEFAULT.copy()
    cfg["outdir"] = args.outdir
    os.makedirs(cfg["outdir"], exist_ok=True)

    print("=" * 78)
    print("VALIDATION SWAPS: REPLICATION")
    print("=" * 78)
    print(f"  n={cfg['n']}, n_val={cfg['n_val']}, "
          f"n_sims_main={cfg['n_sims_main']}, "
          f"n_sims_sens={cfg['n_sims_sens']}, "
          f"delta={cfg['delta']}, seed={cfg['seed']}")
    print(f"  Output: {cfg['outdir']}/")
    t0 = time.time()

    # Figures
    figure_swap_paths(cfg)
    figure_sim_anatomy(cfg)

    # Tables
    all_res = table_main(cfg)
    figure_bias_bars(all_res, cfg)
    table_precision(cfg)
    table_audit_threshold(cfg)
    table_val_size(cfg)

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    print(f"\nDone in {m}m {s}s. All outputs in {cfg['outdir']}/")
    print("Files:")
    for f in sorted(os.listdir(cfg["outdir"])):
        print(f"  {f}")


if __name__ == "__main__":
    main()
