"""Conditional row-correction diagnostics for two pinned replication archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swaps-matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from covariate_swaps import fit_ols  # noqa: E402
from replicate_covariates import write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parent
BUDGETS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
METHODS = ("singleton", "paired_score", "feature_derivative")
LABELS = ("Sum of isolated corrections", "Paired score", "Feature derivative")
COLORS = ("#D55E00", "#0072B2", "#009E73")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(data: Path) -> None:
    """Fetch checksum-pinned inputs, then reproduce the archive preprocessing."""
    manifest = json.loads((ROOT / "research_notes/archives/sources.json").read_text())
    raw = data / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        path = raw / entry["name"]
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--max-time",
                    "180",
                    "--silent",
                    "--show-error",
                    entry["url"],
                    "--output",
                    str(temporary),
                ],
                check=True,
            )
            if sha256(temporary) != entry["sha256"]:
                raise ValueError(f"source checksum mismatch: {entry['name']}")
            temporary.replace(path)
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"source checksum mismatch: {entry['name']}")
    subprocess.run(
        [
            "Rscript",
            str(ROOT / "prepare_archives.R"),
            str(raw),
            str(data / "processed"),
        ],
        check=True,
    )


def load_design(path: Path, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if dataset == "housing":
        proxy = np.column_stack(
            [data["income"], data["proxy_nightlights"], data["proxy_roadlength"]]
        )
        reference = np.column_stack(
            [
                data["income"],
                data["reference_nightlights"],
                data["reference_roadlength"],
            ]
        )
        center, scale = proxy.mean(axis=0), proxy.std(axis=0)
        if np.any(scale == 0):
            raise ValueError("constant housing regressor")
        proxy, reference = (proxy - center) / scale, (reference - center) / scale
    elif dataset == "reddit":
        proxy = data["proxy_incivility"][:, None]
        reference = data["reference_incivility"][:, None]
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    y = data["y"]
    if not all(np.isfinite(a).all() for a in (proxy, reference, y)):
        raise ValueError("missing or nonfinite archive inputs")
    intercept = np.ones(len(y))
    return (
        np.column_stack([intercept, proxy]),
        np.column_stack([intercept, reference]),
        y,
    )


def row_scores(x: np.ndarray, z: np.ndarray, y: np.ndarray, target: int) -> dict:
    """Exact single-row OLS updates and the two fixed additive approximations."""
    beta, _ = fit_ols(x, y)
    delta = z - x
    gram = x.T @ x
    residual = y - x @ beta
    change = delta @ beta
    linear = delta * residual[:, None] - x * change[:, None]
    finite = linear - delta * change[:, None]
    direction = np.linalg.solve(gram, np.eye(x.shape[1])[:, target])
    updated = gram + z[:, :, None] * z[:, None, :] - x[:, :, None] * x[:, None, :]
    eigenvalues = np.linalg.eigvalsh(updated)
    if np.any(eigenvalues[:, 0] <= np.finfo(float).eps * eigenvalues[:, -1]):
        raise ValueError("a singleton update is singular or numerically unstable")
    singleton = np.linalg.solve(updated, finite[..., None])[..., 0][:, target]
    return {
        "singleton": singleton,
        "paired_score": finite @ direction,
        "feature_derivative": linear @ direction,
    }


def hc0_scale(x: np.ndarray, y: np.ndarray, target: int) -> float:
    """HC0 scale for descriptive normalization, without a population CI claim."""
    beta, _ = fit_ols(x, y)
    direction = np.linalg.solve(x.T @ x, np.eye(x.shape[1])[:, target])
    return float(np.linalg.norm((y - x @ beta) * (x @ direction)))


def check_singletons(x, z, y, scores, seed: int) -> dict:
    beta, _ = fit_ols(x, y)
    rng = np.random.default_rng(seed)
    leverage = np.sum(x * np.linalg.solve(x.T @ x, x.T).T, axis=1)
    update = np.linalg.norm(z - x, axis=1)
    indices = np.unique(
        np.concatenate(
            [
                rng.choice(len(y), size=min(100, len(y)), replace=False),
                np.argsort(leverage)[-20:],
                np.argsort(update)[-20:],
            ]
        )
    )
    discrepancies = []
    for i in indices:
        corrected = x.copy()
        corrected[i] = z[i]
        actual = fit_ols(corrected, y)[0][1] - beta[1]
        discrepancies.append(abs(actual - scores["singleton"][i]))
        np.testing.assert_allclose(
            actual, scores["singleton"][i], atol=1e-10, rtol=1e-7
        )
    return {"checked": len(indices), "max_absolute_discrepancy": max(discrepancies)}


def summarize(rows: list[dict], scale: float) -> list[dict]:
    summaries = []
    for count in sorted({r["corrected_rows"] for r in rows}):
        group = [r for r in rows if r["corrected_rows"] == count]
        actual = np.array([r["coefficient"] for r in group])
        m = len(group)
        for method in METHODS:
            prediction = np.array([r[method] for r in group])
            errors = prediction - actual
            rmse = float(np.sqrt(np.mean(errors**2)))
            mcse = (
                float(np.std(errors**2, ddof=1) / (2 * rmse * np.sqrt(m)))
                if m > 1 and rmse > 0
                else 0.0
            )
            summaries.append(
                {
                    "dataset": group[0]["dataset"],
                    "corrected_rows": count,
                    "fraction": group[0]["fraction"],
                    "sampled_sets": m,
                    "method": method,
                    "refit_mean": float(actual.mean()),
                    "refit_sd": float(actual.std(ddof=1)) if m > 1 else 0.0,
                    "refit_mean_mcse": (
                        float(actual.std(ddof=1) / np.sqrt(m)) if m > 1 else 0.0
                    ),
                    "refit_q05": float(np.quantile(actual, 0.05)),
                    "refit_q95": float(np.quantile(actual, 0.95)),
                    "approximation_mean": float(prediction.mean()),
                    "mean_signed_error": float(errors.mean()),
                    "rmse": rmse,
                    "rmse_mcse": mcse,
                    "sampled_max_absolute_error": float(np.max(np.abs(errors))),
                    "rmse_in_hc0_units": rmse / scale,
                    "fraction_error_over_0.1_hc0": float(
                        np.mean(np.abs(errors) > 0.1 * scale)
                    ),
                    "fraction_error_over_0.5_hc0": float(
                        np.mean(np.abs(errors) > 0.5 * scale)
                    ),
                    "max_design_condition": max(r["condition"] for r in group),
                }
            )
    return summaries


def analyze(dataset: str, path: Path, draws: int, seed: int) -> tuple:
    x, z, y = load_design(path, dataset)
    base, base_condition = fit_ols(x, y)
    full, full_condition = fit_ols(z, y)
    scores = row_scores(x, z, y, target=1)
    checks = check_singletons(x, z, y, scores, seed + 1)
    scale = hc0_scale(z, y, target=1)
    n = len(y)
    rng = np.random.default_rng(seed)
    rows, failures = [], []

    def record(indices, draw, beta, condition):
        row = {
            "dataset": dataset,
            "draw": draw,
            "corrected_rows": len(indices),
            "fraction": len(indices) / n,
            "coefficient": float(beta),
            "condition": condition,
            "set_sha256": hashlib.sha256(
                np.sort(indices).astype("<i8").tobytes()
            ).hexdigest(),
        }
        row.update(
            {
                method: float(base[1] + scores[method][indices].sum())
                for method in METHODS
            }
        )
        rows.append(row)

    for draw in range(draws):
        ordering = rng.permutation(n)
        for fraction in BUDGETS[:-1]:
            indices = ordering[: max(1, round(fraction * n))]
            corrected = x.copy()
            corrected[indices] = z[indices]
            try:
                beta, condition = fit_ols(corrected, y)
                record(indices, draw, beta[1], condition)
            except (ValueError, np.linalg.LinAlgError) as error:
                failures.append(
                    {
                        "dataset": dataset,
                        "draw": draw,
                        "fraction": fraction,
                        "error": str(error),
                    }
                )
    record(np.arange(n), -1, full[1], full_condition)
    shift = float(full[1] - base[1])
    endpoint = {
        "rows": n,
        "changed_rows": int(np.any(x != z, axis=1).sum()),
        "proxy_coefficient": float(base[1]),
        "reference_coefficient": float(full[1]),
        "endpoint_shift": shift,
        "reference_hc0_scale": scale,
        "proxy_design_condition": base_condition,
        "reference_design_condition": full_condition,
        "singleton_checks": checks,
        "approximations": {
            method: {
                "coefficient": rows[-1][method],
                "signed_error": rows[-1][method] - full[1],
                "absolute_error_over_endpoint_shift": (
                    abs((rows[-1][method] - full[1]) / shift) if shift else None
                ),
            }
            for method in METHODS
        },
    }
    return rows, summarize(rows, scale), endpoint, failures


def figures(out: Path, summaries: list[dict], endpoints: dict) -> None:
    plot_labels = ("Isolated sum", "Paired score", "Feature derivative")
    plt.rcParams.update(
        {"font.size": 13, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, dataset in zip(axes, ("housing", "reddit"), strict=True):
        base = endpoints[dataset]["proxy_coefficient"]
        group = [
            r
            for r in summaries
            if r["dataset"] == dataset and r["method"] == "singleton"
        ]
        p = [0] + [r["fraction"] for r in group]
        means = [base] + [r["refit_mean"] for r in group]
        ax.fill_between(
            p,
            [base] + [r["refit_q05"] for r in group],
            [base] + [r["refit_q95"] for r in group],
            color="0.85",
            label="Refit 5th–95th percentiles",
        )
        ax.plot(p, means, color="black", marker="o", markersize=3, label="Mean refit")
        for method, label, color in zip(METHODS, plot_labels, COLORS, strict=True):
            g = [
                r
                for r in summaries
                if r["dataset"] == dataset and r["method"] == method
            ]
            ax.plot(
                p,
                [base] + [r["approximation_mean"] for r in g],
                linestyle="--",
                color=color,
                label=label,
            )
        ax.set(title=dataset.capitalize(), xlabel="Fraction of rows corrected")
        ax.set_ylabel(
            "Income coefficient (per SD)"
            if dataset == "housing"
            else "Incivility coefficient (score points)"
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    fig.savefig(out / "archive_paths.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, dataset in zip(axes, ("housing", "reddit"), strict=True):
        for method, label, color in zip(METHODS, plot_labels, COLORS, strict=True):
            g = [
                r
                for r in summaries
                if r["dataset"] == dataset and r["method"] == method
            ]
            ax.plot(
                [r["fraction"] for r in g],
                [r["rmse_in_hc0_units"] for r in g],
                marker="o",
                markersize=3,
                color=color,
                label=label,
            )
        ax.set(
            title=dataset.capitalize(),
            xlabel="Fraction of rows corrected",
            ylabel="Approximation RMSE / reference HC0 scale",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out / "archive_errors.pdf", bbox_inches="tight")
    plt.close(fig)


def manuscript_inputs(out: Path, endpoints: dict) -> None:
    columns = [endpoints[name] for name in ("housing", "reddit")]
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r" & Housing & Reddit \\",
        r"\midrule",
    ]
    lines.append("Rows & " + " & ".join(f"{e['rows']:,}" for e in columns) + r" \\")
    for label, key in (
        ("Proxy coefficient", "proxy_coefficient"),
        ("Reference coefficient", "reference_coefficient"),
        ("Actual correction", "endpoint_shift"),
    ):
        lines.append(
            label + " & " + " & ".join(f"{e[key]:.5f}" for e in columns) + r" \\"
        )
    for method, label in zip(METHODS, LABELS, strict=True):
        values = [
            e["approximations"][method]["coefficient"] - e["proxy_coefficient"]
            for e in columns
        ]
        lines.append(label + " & " + " & ".join(f"{v:.5f}" for v in values) + r" \\")
    values = [
        abs(e["approximations"]["singleton"]["signed_error"]) / e["reference_hc0_scale"]
        for e in columns
    ]
    lines.append(
        r"$|R(N)|$/reference HC0 scale & "
        + " & ".join(f"{v:.2f}" for v in values)
        + r" \\"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "table.tex").write_text("\n".join(lines) + "\n")
    housing, reddit = columns
    housing_ratio = (
        housing["approximations"]["singleton"]["coefficient"]
        - housing["proxy_coefficient"]
    ) / housing["endpoint_shift"]
    reddit_fraction = reddit["approximations"]["singleton"][
        "absolute_error_over_endpoint_shift"
    ]
    (out / "macros.tex").write_text(
        rf"\newcommand{{\ArchiveHousingRatio}}{{{housing_ratio:.1f}}}"
        + "\n"
        + rf"\newcommand{{\ArchiveRedditUnderstatement}}{{{100 * reddit_fraction:.0f}}}"
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/archives")
    parser.add_argument("--outdir", type=Path, default=ROOT / "generated/archives")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if args.draws < 2:
        parser.error("at least two draws are needed for Monte Carlo summaries")
    if args.prepare:
        prepare(args.data)
    args.outdir.mkdir(parents=True, exist_ok=True)
    processed = args.data / "processed"
    all_rows, all_summaries, all_failures, endpoints = [], [], [], {}
    for dataset in ("housing", "reddit"):
        rows, summaries, endpoint, failures = analyze(
            dataset, processed / f"{dataset}.csv", args.draws, args.seed
        )
        all_rows.extend(rows)
        all_summaries.extend(summaries)
        all_failures.extend(failures)
        endpoints[dataset] = endpoint
    write_csv(args.outdir / "sampled_sets.csv", all_rows)
    write_csv(args.outdir / "summary.csv", all_summaries)
    (args.outdir / "endpoints.json").write_text(json.dumps(endpoints, indent=2) + "\n")
    (args.outdir / "failures.json").write_text(
        json.dumps(all_failures, indent=2) + "\n"
    )
    metadata = {
        "seed": args.seed,
        "draws": args.draws,
        "budgets": BUDGETS,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "input_sha256": {
            p.name: sha256(p) for p in sorted(processed.iterdir()) if p.is_file()
        },
        "source_manifest_sha256": sha256(ROOT / "research_notes/archives/sources.json"),
        "code_sha256": {
            p: sha256(ROOT / p)
            for p in (
                "replicate_archives.py",
                "prepare_archives.R",
                "covariate_swaps.py",
            )
        },
    }
    (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for filename in ("R-session.txt", "preparation_audit.csv"):
        shutil.copy2(processed / filename, args.outdir / filename)
    figures(args.outdir, all_summaries, endpoints)
    manuscript_inputs(args.outdir, endpoints)
    print(json.dumps(endpoints, indent=2))
    if all_failures:
        raise RuntimeError("Some correction sets failed; see failures.json")


if __name__ == "__main__":
    main()
