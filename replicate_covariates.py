"""Reproduce the declared finite covariate-correction benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/swaps-matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from covariate_swaps import (  # noqa: E402
    ExactGame,
    additive_values,
    enumerate_game,
    make_benchmark,
)

SCENARIOS = ("diffuse", "concentrated", "aligned")
LABELS = {
    "diffuse": "Diffuse",
    "concentrated": "One active batch",
    "aligned": "Treatment-aligned",
}
METHODS = ("paired_score", "singleton", "feature_derivative")
METHOD_LABELS = ("Paired score", "Single-repair sum", "Feature derivative")
COLORS = ("#0072B2", "#D55E00", "#009E73")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(scenario: str, seed: int, rmse: float, n: int, players: int) -> dict:
    game = make_benchmark(scenario, seed, rmse, n, players)
    exact = enumerate_game(game)
    scores = game.additive_scores()
    approximations = {
        key: additive_values(exact.values[0], value) for key, value in scores.items()
    }
    p = np.linspace(0, 1, 101)
    mean, sd = exact.bernoulli(p)
    gap = exact.values[-1] - exact.values[0]
    row = {
        "scenario": scenario,
        "seed": seed,
        "rmse": rmse,
        "proxy_coefficient": exact.values[0],
        "verified_coefficient": exact.values[-1],
        "endpoint_gap": gap,
        "max_path_departure": np.max(np.abs(mean - (exact.values[0] + p * gap))),
        "half_budget_sd": np.std(exact.fixed_size()[players // 2]),
        "max_design_condition": exact.condition_numbers.max(),
        "shapley_efficiency_error": abs(exact.shapley().sum() - gap),
        "banzhaf_efficiency_error": abs(exact.banzhaf().sum() - gap),
    }
    for key, values in approximations.items():
        difference = values - exact.values
        row[f"{key}_rmse"] = np.sqrt(np.mean(difference**2))
        row[f"{key}_max_error"] = np.max(np.abs(difference))
    return dict(
        game=game,
        exact=exact,
        scores=scores,
        approximations=approximations,
        p=p,
        mean=mean,
        sd=sd,
        summary=row,
    )


def save_figure(fig, outdir: Path, name: str) -> None:
    fig.savefig(
        outdir / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    fig.savefig(outdir / f"{name}.png", dpi=180)
    plt.close(fig)


def main_figures(results: list[dict], outdir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(8.5, 5.5), sharex=True, layout="constrained"
    )
    for j, result in enumerate(results):
        p, exact = result["p"], result["exact"]
        axes[0, j].plot(
            p, result["mean"], color="black", label="Exact mean", linewidth=2
        )
        approximate = ExactGame(
            result["approximations"]["paired_score"], exact.condition_numbers
        )
        approx_mean, approx_sd = approximate.bernoulli(p)
        axes[0, j].plot(
            p, approx_mean, color=COLORS[0], linestyle="--", label="Paired-score mean"
        )
        axes[0, j].plot(
            p,
            exact.values[0] + p * (exact.values[-1] - exact.values[0]),
            color="#777777",
            linestyle=":",
            label="Endpoint interpolation",
        )
        axes[0, j].axhline(exact.values[-1], color="#999999", linewidth=0.8)
        axes[0, j].set_title(LABELS[result["summary"]["scenario"]])
        axes[1, j].plot(
            p, result["sd"], color="black", linewidth=2, label="Bernoulli SD"
        )
        axes[1, j].plot(
            p, approx_sd, color=COLORS[0], linestyle="--", label="Paired-score SD"
        )
        groups = exact.fixed_size()
        axes[1, j].plot(
            np.arange(exact.size + 1) / exact.size,
            [np.std(group) for group in groups],
            color=COLORS[1],
            marker="o",
            markersize=3,
            label="Fixed-budget SD",
        )
        axes[1, j].set_xlabel("Correction probability, p")
        axes[1, j].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("Treatment coefficient")
    axes[1, 0].set_ylabel("SD across correction sets")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside upper center",
        ncol=3,
        fontsize=9.5,
        frameon=False,
    )
    fig.legend(
        *axes[1, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        fontsize=9.5,
        frameon=False,
    )
    save_figure(fig, outdir, "correction_paths")

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.4), layout="constrained")
    for ax, result in zip(axes, results, strict=True):
        exact = result["exact"]
        x = np.arange(1, exact.size + 1)
        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.plot(x, exact.shapley(), "o", color="black", label="Shapley")
        ax.plot(
            x - 0.12,
            result["scores"]["paired_score"],
            "x",
            color=COLORS[0],
            label="Paired score",
        )
        ax.plot(
            x + 0.12,
            result["scores"]["singleton"],
            "+",
            color=COLORS[1],
            label="Single repair",
        )
        ax.set_title(LABELS[result["summary"]["scenario"]])
        ax.set_xlabel("Correction batch")
        ax.set_xticks(x)
    axes[0].set_ylabel("Attributed coefficient change")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside upper center",
        ncol=3,
        fontsize=9.5,
        frameon=False,
    )
    save_figure(fig, outdir, "attributions")


def repeat_figure(rows: list[dict], outdir: Path) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(8.5, 3.6), sharey=True, layout="constrained"
    )
    levels = sorted({row["rmse"] for row in rows})
    for ax, scenario in zip(axes, SCENARIOS, strict=True):
        for key, label, color in zip(METHODS, METHOD_LABELS, COLORS, strict=True):
            values = [
                np.array(
                    [
                        row[f"{key}_rmse"]
                        for row in rows
                        if row["scenario"] == scenario and row["rmse"] == level
                    ]
                )
                for level in levels
            ]
            medians = [np.median(v) for v in values]
            if max(np.max(v) for v in values) < 1e-12:
                ax.text(
                    0.04,
                    0.05,
                    f"{label}: exact",
                    transform=ax.transAxes,
                    color=color,
                    fontsize=9,
                )
                continue
            ax.plot(
                levels,
                medians,
                "o-",
                color=color,
                label=label,
                markersize=3,
            )
            ax.fill_between(
                levels,
                [np.quantile(v, 0.1) for v in values],
                [np.quantile(v, 0.9) for v in values],
                color=color,
                alpha=0.12,
            )
        ax.set_title(LABELS[scenario])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Proxy RMSE")
        ax.set_xticks(levels, [str(x) for x in levels])
    axes[0].set_ylabel("Coalition approximation RMSE")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside upper center",
        ncol=3,
        fontsize=9.5,
        frameon=False,
    )
    save_figure(fig, outdir, "approximation_benchmark")


def write_tables(results: list[dict], repeated: list[dict], outdir: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Error pattern & Proxy $\hat\tau$ & Gold $\hat\tau$ & Curvature & "
        + rf"SD at $k={results[0]['exact'].size // 2}$ & IF RMSE \\",
        r"\midrule",
    ]
    for result in results:
        row = result["summary"]
        columns = [
            row[key]
            for key in (
                "proxy_coefficient",
                "verified_coefficient",
                "max_path_departure",
                "half_budget_sd",
                "paired_score_rmse",
            )
        ]
        lines.append(
            LABELS[row["scenario"]]
            + " & "
            + " & ".join(f"{value:.3f}" for value in columns)
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (outdir / "table_main.tex").write_text("\n".join(lines) + "\n")
    summaries = []
    for scenario in SCENARIOS:
        for level in sorted({row["rmse"] for row in repeated}):
            selected = [
                row
                for row in repeated
                if row["scenario"] == scenario and row["rmse"] == level
            ]
            for method in METHODS:
                errors = np.array([row[f"{method}_rmse"] for row in selected])
                summaries.append(
                    dict(
                        scenario=scenario,
                        rmse=level,
                        method=method,
                        median=np.median(errors),
                        q10=np.quantile(errors, 0.1),
                        q90=np.quantile(errors, 0.9),
                    )
                )
    write_csv(outdir / "benchmark_summary.csv", summaries)
    max_condition = max(row["max_design_condition"] for row in repeated)
    macros = {
        "CovariateReplications": len(repeated),
        "CovariateMaxCondition": f"{max_condition:.2f}",
    }
    for result in results:
        row = result["summary"]
        prefix = row["scenario"].title()
        for name, key in (
            ("Gap", "endpoint_gap"),
            ("Curvature", "max_path_departure"),
            ("IFError", "paired_score_rmse"),
            ("HalfSD", "half_budget_sd"),
        ):
            macros[prefix + name] = f"{row[key]:.3f}"
    (outdir / "macros.tex").write_text(
        "".join(
            "\\newcommand{\\" + key + "}{" + str(value) + "}\n"
            for key, value in macros.items()
        )
    )


def run(outdir: Path, fast: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    n, players, seeds = (48, 6, range(3)) if fast else (96, 8, range(20))
    results = [evaluate(scenario, 2024, 0.75, n, players) for scenario in SCENARIOS]
    paths, fixed, attribution, coalitions = [], [], [], []
    for result in results:
        scenario = result["summary"]["scenario"]
        exact = result["exact"]
        for p, mean, sd in zip(result["p"], result["mean"], result["sd"], strict=True):
            paths.append(dict(scenario=scenario, p=p, mean=mean, sd=sd))
        for k, values in enumerate(exact.fixed_size()):
            fixed.append(
                dict(
                    scenario=scenario,
                    k=k,
                    count=len(values),
                    mean=np.mean(values),
                    sd=np.std(values),
                    q05=np.quantile(values, 0.05),
                    median=np.median(values),
                    q95=np.quantile(values, 0.95),
                )
            )
        for j in range(players):
            attribution.append(
                dict(
                    scenario=scenario,
                    player=j,
                    shapley=exact.shapley()[j],
                    banzhaf=exact.banzhaf()[j],
                    **{key: value[j] for key, value in result["scores"].items()},
                )
            )
        for mask, value in enumerate(exact.values):
            coalitions.append(
                dict(
                    scenario=scenario,
                    mask=mask,
                    k=mask.bit_count(),
                    value=value,
                    condition=exact.condition_numbers[mask],
                    **{
                        key: values[mask]
                        for key, values in result["approximations"].items()
                    },
                )
            )
    repeated, failures = [], []
    for seed in seeds:
        for level in (0.05, 0.25, 0.75, 1.5):
            for scenario in SCENARIOS:
                try:
                    repeated.append(
                        evaluate(scenario, seed, level, n, players)["summary"]
                    )
                except (ValueError, np.linalg.LinAlgError) as error:
                    failures.append(
                        dict(seed=seed, rmse=level, scenario=scenario, error=str(error))
                    )
    (outdir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if repeated:
        write_csv(outdir / "benchmark.csv", repeated)
    if failures:
        (outdir / "metadata.json").write_text(
            json.dumps(
                dict(
                    status="failed",
                    failures=len(failures),
                    successful_designs=len(repeated),
                ),
                indent=2,
            )
            + "\n"
        )
        raise RuntimeError(
            f"{len(failures)} benchmark designs failed; see failures.json"
        )
    for filename, rows in (
        ("main", [r["summary"] for r in results]),
        ("paths", paths),
        ("fixed_budgets", fixed),
        ("attributions", attribution),
        ("coalitions", coalitions),
        ("benchmark", repeated),
    ):
        write_csv(outdir / f"{filename}.csv", rows)
    write_tables(results, repeated, outdir)
    main_figures(results, outdir)
    repeat_figure(repeated, outdir)
    metadata = dict(
        n=n,
        players=players,
        main_seed=2024,
        main_rmse=0.75,
        repeated_seeds=list(seeds),
        levels=[0.05, 0.25, 0.75, 1.5],
        scenarios=list(SCENARIOS),
        enumeration="all correction subsets",
        error_weighting="uniform over all subsets (Bernoulli p=0.5)",
        condition_number="design matrix 2-norm",
        failures=0,
        fast=fast,
    )
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Wrote {len(results)} main games and "
        f"{len(repeated)} repeated games to {outdir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("generated/covariates"))
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    run(args.outdir, args.fast)
