"""Generate the validation-swap distributions, tables, and figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


SCENARIOS = ("diffuse", "concentrated", "offsetting")
SCENARIO_LABELS = {
    "diffuse": "Diffuse",
    "concentrated": "Concentrated",
    "offsetting": "Offsetting",
}
TARGET_ATE = 0.10
TOTAL_DISTORTION = 0.04
TARGET_RMSE = 0.20
TAIL_THRESHOLD = 0.05

DEFAULT: dict[str, Any] = {
    "clusters": 20,
    "cluster_size": 50,
    "seed": 2024,
}

FAST: dict[str, Any] = {
    **DEFAULT,
    "clusters": 16,
    "cluster_size": 30,
}


@dataclass(frozen=True)
class Experiment:
    gold: np.ndarray
    proxy: np.ndarray
    treatment: np.ndarray
    cluster: np.ndarray
    contributions: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast", action="store_true", help="Use a smaller exact design"
    )
    parser.add_argument("--outdir", default="generated", help="Output directory")
    parser.add_argument("--clusters", type=int, help="Number of experimental clusters")
    parser.add_argument("--cluster-size", type=int, help="Observations per cluster")
    parser.add_argument("--seed", type=int, help="Base random seed")
    return parser.parse_args()


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    cfg = FAST.copy() if args.fast else DEFAULT.copy()
    overrides = {
        "clusters": args.clusters,
        "cluster_size": args.cluster_size,
        "seed": args.seed,
    }
    cfg.update({key: value for key, value in overrides.items() if value is not None})
    if cfg["clusters"] < 4 or cfg["clusters"] % 2:
        raise ValueError("clusters must be an even integer of at least four")
    if cfg["cluster_size"] < 2:
        raise ValueError("cluster-size must be at least two")
    cfg["outdir"] = args.outdir
    return cfg


def difference_in_means(outcome: np.ndarray, treatment: np.ndarray) -> float:
    return float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean())


def desired_contributions(scenario: str, clusters: int) -> np.ndarray:
    """Set cluster contributions while holding the endpoint gap fixed when requested."""
    values = np.zeros(clusters)
    half = clusters // 2
    if scenario == "diffuse":
        values.fill(TOTAL_DISTORTION / clusters)
    elif scenario == "concentrated":
        values[0] = TOTAL_DISTORTION / 2
        values[half] = TOTAL_DISTORTION / 2
    elif scenario == "offsetting":
        values[0] = TOTAL_DISTORTION / 2
        values[1] = -TOTAL_DISTORTION / 2
        values[half] = TOTAL_DISTORTION / 2
        values[half + 1] = -TOTAL_DISTORTION / 2
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return values


def cluster_contributions(
    gold: np.ndarray,
    proxy: np.ndarray,
    treatment: np.ndarray,
    cluster: np.ndarray,
) -> np.ndarray:
    """Allocate the gold-minus-proxy ATE gap across experimental clusters."""
    residual = gold - proxy
    treated_n = int(np.sum(treatment == 1))
    control_n = int(np.sum(treatment == 0))
    values = np.empty(int(cluster.max()) + 1)
    for group in range(len(values)):
        rows = cluster == group
        if np.unique(treatment[rows]).size != 1:
            raise ValueError("treatment must be constant within cluster")
        if treatment[rows][0] == 1:
            values[group] = residual[rows].sum() / treated_n
        else:
            values[group] = -residual[rows].sum() / control_n
    return values


def make_experiment(
    scenario: str,
    clusters: int = 20,
    cluster_size: int = 50,
    seed: int = 2024,
) -> Experiment:
    """Create a clustered experiment with matched proxy RMSE across scenarios."""
    if clusters < 4 or clusters % 2:
        raise ValueError("clusters must be an even integer of at least four")
    if cluster_size < 2:
        raise ValueError("cluster_size must be at least two")

    rng = np.random.default_rng(seed)
    cluster = np.repeat(np.arange(clusters), cluster_size)
    treatment_by_cluster = np.r_[
        np.ones(clusters // 2, dtype=int),
        np.zeros(clusters // 2, dtype=int),
    ]
    treatment = treatment_by_cluster[cluster]

    cluster_effect = rng.normal(0.0, 0.35, clusters)[cluster]
    individual_noise = rng.normal(0.0, 1.0, len(cluster))
    baseline = cluster_effect + individual_noise
    for arm in (0, 1):
        baseline[treatment == arm] -= baseline[treatment == arm].mean()
    gold = TARGET_ATE * treatment + baseline

    target_contributions = desired_contributions(scenario, clusters)
    units_per_arm = len(gold) // 2
    cluster_mean_error = np.empty(clusters)
    for group in range(clusters):
        sign = 1.0 if treatment_by_cluster[group] == 1 else -1.0
        cluster_mean_error[group] = (
            sign * target_contributions[group] * units_per_arm / cluster_size
        )

    centered_noise = rng.normal(size=len(gold))
    for group in range(clusters):
        rows = cluster == group
        centered_noise[rows] -= centered_noise[rows].mean()
    mean_component = cluster_mean_error[cluster]
    remaining_mse = TARGET_RMSE**2 - float(np.mean(mean_component**2))
    if remaining_mse < 0:
        raise ValueError("target RMSE is too small for the requested contributions")
    noise_scale = math.sqrt(remaining_mse / float(np.mean(centered_noise**2)))
    scoring_error = mean_component + noise_scale * centered_noise
    proxy = gold - scoring_error
    contributions = cluster_contributions(gold, proxy, treatment, cluster)

    if not np.allclose(contributions, target_contributions, atol=1e-12):
        raise RuntimeError("constructed contributions do not match their targets")
    if not np.isclose(np.sqrt(np.mean(scoring_error**2)), TARGET_RMSE):
        raise RuntimeError("constructed scoring error does not match target RMSE")
    return Experiment(gold, proxy, treatment, cluster, contributions)


def exact_subset_sums(values: np.ndarray) -> list[np.ndarray]:
    """Return every fixed-size subset sum, indexed by subset size."""
    distributions = [np.array([0.0])] + [np.empty(0) for _ in values]
    for processed, value in enumerate(values, start=1):
        for size in range(processed, 0, -1):
            added = distributions[size - 1] + value
            distributions[size] = np.concatenate([distributions[size], added])
    for size, distribution in enumerate(distributions):
        expected = math.comb(len(values), size)
        if len(distribution) != expected:
            raise RuntimeError("subset enumeration failed")
    return distributions


def raw_swap_distributions(experiment: Experiment) -> list[np.ndarray]:
    proxy_ate = difference_in_means(experiment.proxy, experiment.treatment)
    return [proxy_ate + sums for sums in exact_subset_sums(experiment.contributions)]


def scaled_swap_distributions(experiment: Experiment) -> list[np.ndarray]:
    """Return design-scaled estimates for every nonempty validation set size."""
    proxy_ate = difference_in_means(experiment.proxy, experiment.treatment)
    clusters = len(experiment.contributions)
    subset_sums = exact_subset_sums(experiment.contributions)
    distributions = [np.array([np.nan])]
    for size in range(1, clusters + 1):
        distributions.append(proxy_ate + (clusters / size) * subset_sums[size])
    return distributions


def finite_population_variance(values: np.ndarray, size: int, scaled: bool) -> float:
    """Variance of a fixed-size simple-random-sample sum or scaled total."""
    clusters = len(values)
    if not 1 <= size <= clusters:
        raise ValueError("size must lie between one and the number of clusters")
    population_variance = float(np.var(values, ddof=1))
    raw_variance = size * (1 - size / clusters) * population_variance
    if scaled:
        return (clusters / size) ** 2 * raw_variance
    return raw_variance


def prediction_metrics(experiment: Experiment) -> tuple[float, float]:
    error = experiment.gold - experiment.proxy
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = float(np.sum((experiment.gold - experiment.gold.mean()) ** 2))
    r_squared = 1 - float(error @ error) / denominator
    return rmse, r_squared


def make_experiments(cfg: dict[str, Any]) -> dict[str, Experiment]:
    return {
        scenario: make_experiment(
            scenario,
            clusters=cfg["clusters"],
            cluster_size=cfg["cluster_size"],
            seed=cfg["seed"],
        )
        for scenario in SCENARIOS
    }


def budget_grid(clusters: int) -> list[int]:
    candidates = {1, 2, clusters // 4, clusters // 2, 3 * clusters // 4, clusters}
    return sorted(size for size in candidates if 1 <= size <= clusters)


def scenario_table(
    cfg: dict[str, Any], experiments: dict[str, Experiment]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario, experiment in experiments.items():
        gold_ate = difference_in_means(experiment.gold, experiment.treatment)
        proxy_ate = difference_in_means(experiment.proxy, experiment.treatment)
        rmse, r_squared = prediction_metrics(experiment)
        absolute = np.abs(experiment.contributions)
        rows.append(
            {
                "scenario": scenario,
                "gold_ate": gold_ate,
                "proxy_ate": proxy_ate,
                "endpoint_gap": gold_ate - proxy_ate,
                "rmse": rmse,
                "r_squared": r_squared,
                "contribution_sd": float(np.std(experiment.contributions, ddof=1)),
                "top_two_absolute_share": float(
                    np.sort(absolute)[-2:].sum() / absolute.sum()
                    if absolute.sum() > 0
                    else 0.0
                ),
            }
        )
    write_csv(
        Path(cfg["outdir"]) / "table_scenarios.csv",
        list(rows[0]),
        rows,
    )
    return rows


def budget_table(
    cfg: dict[str, Any], experiments: dict[str, Experiment]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario, experiment in experiments.items():
        gold_ate = difference_in_means(experiment.gold, experiment.treatment)
        distributions = scaled_swap_distributions(experiment)
        for size in budget_grid(cfg["clusters"]):
            estimates = distributions[size]
            rows.append(
                {
                    "scenario": scenario,
                    "validated_clusters": size,
                    "mean": float(np.mean(estimates)),
                    "bias": float(np.mean(estimates) - gold_ate),
                    "sd": float(np.std(estimates, ddof=0)),
                    "p05": float(np.quantile(estimates, 0.05)),
                    "median": float(np.quantile(estimates, 0.50)),
                    "p95": float(np.quantile(estimates, 0.95)),
                    "tail_probability": float(
                        np.mean(np.abs(estimates - gold_ate) > TAIL_THRESHOLD)
                    ),
                    "sign_reversal_probability": float(
                        np.mean(np.sign(estimates) != np.sign(gold_ate))
                    ),
                }
            )
    write_csv(
        Path(cfg["outdir"]) / "table_budgets.csv",
        list(rows[0]),
        rows,
    )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def format_percent(value: float, digits: int = 0) -> str:
    return f"{100 * value:.{digits}f}\\%"


def latex_scenario_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Error pattern & Gold ATE & Proxy ATE & Gap & RMSE & $R^2$ & Top-two share \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{SCENARIO_LABELS[row['scenario']]} & "
            f"{format_number(row['gold_ate'])} & "
            f"{format_number(row['proxy_ate'])} & "
            f"{format_number(row['endpoint_gap'])} & "
            f"{format_number(row['rmse'])} & "
            f"{format_number(row['r_squared'])} & "
            f"{format_percent(row['top_two_absolute_share'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_budget_table(path: Path, rows: list[dict[str, Any]], clusters: int) -> None:
    selected = {max(1, clusters // 10), clusters // 4, clusters // 2, clusters}
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Error pattern & Gold clusters & Mean & SD & 5th & 95th & Tail risk \\",
        r"\midrule",
    ]
    previous = ""
    for row in rows:
        if row["validated_clusters"] not in selected:
            continue
        if previous and row["scenario"] != previous:
            lines.append(r"\addlinespace")
        label = SCENARIO_LABELS[row["scenario"]] if row["scenario"] != previous else ""
        lines.append(
            f"{label} & {row['validated_clusters']} & "
            f"{format_number(row['mean'])} & {format_number(row['sd'])} & "
            f"{format_number(row['p05'])} & {format_number(row['p95'])} & "
            f"{format_percent(row['tail_probability'])} \\\\"
        )
        previous = row["scenario"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="0.90", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9)


def figure_same_mean(cfg: dict[str, Any], experiments: dict[str, Experiment]) -> None:
    clusters = cfg["clusters"]
    diffuse = experiments["diffuse"]
    concentrated = experiments["concentrated"]
    raw_diffuse = raw_swap_distributions(diffuse)
    raw_concentrated = raw_swap_distributions(concentrated)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    sizes = np.arange(clusters + 1)
    axes[0].plot(
        sizes,
        [np.mean(values) for values in raw_diffuse],
        color="#0072B2",
        linewidth=2.3,
        label="Diffuse",
    )
    axes[0].plot(
        sizes,
        [np.mean(values) for values in raw_concentrated],
        color="#D55E00",
        linewidth=1.7,
        linestyle="--",
        label="Concentrated",
    )
    axes[0].axhline(TARGET_ATE, color="0.35", linewidth=1, linestyle=":")
    axes[0].set_xlabel("Gold-labeled clusters")
    axes[0].set_ylabel("Mean raw-swap ATE")
    axes[0].set_title("A. The mean paths coincide")
    axes[0].legend(frameon=False, fontsize=9)
    style_axis(axes[0])

    size = max(1, clusters // 4)
    colors = {"Diffuse": "#0072B2", "Concentrated": "#D55E00"}
    for label, experiment, distribution in (
        ("Diffuse", diffuse, raw_diffuse[size]),
        ("Concentrated", concentrated, raw_concentrated[size]),
    ):
        proxy_ate = difference_in_means(experiment.proxy, experiment.treatment)
        movement = np.round(distribution - proxy_ate, 12)
        values, counts = np.unique(movement, return_counts=True)
        probability = counts / counts.sum()
        axes[1].vlines(
            values,
            0,
            probability,
            color=colors[label],
            linewidth=2,
        )
        axes[1].scatter(
            values,
            probability,
            color=colors[label],
            s=28,
            label=label,
            zorder=3,
        )
    axes[1].set_xlabel("Raw-swap ATE movement")
    axes[1].set_ylabel("Exact probability")
    axes[1].set_title(f"B. The {size}-cluster distributions differ")
    axes[1].legend(frameon=False, fontsize=9)
    style_axis(axes[1])
    figure.tight_layout(w_pad=2.4)
    figure.savefig(
        Path(cfg["outdir"]) / "fig1_same_mean.png", dpi=220, bbox_inches="tight"
    )
    plt.close(figure)


def figure_budget_distributions(
    cfg: dict[str, Any], experiments: dict[str, Experiment]
) -> None:
    clusters = cfg["clusters"]
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharex=True, sharey=True)
    sizes = np.arange(1, clusters + 1)
    for axis, scenario in zip(axes, SCENARIOS, strict=True):
        experiment = experiments[scenario]
        distributions = scaled_swap_distributions(experiment)
        q05 = np.array([np.quantile(distributions[size], 0.05) for size in sizes])
        q25 = np.array([np.quantile(distributions[size], 0.25) for size in sizes])
        q50 = np.array([np.quantile(distributions[size], 0.50) for size in sizes])
        q75 = np.array([np.quantile(distributions[size], 0.75) for size in sizes])
        q95 = np.array([np.quantile(distributions[size], 0.95) for size in sizes])
        gold_ate = difference_in_means(experiment.gold, experiment.treatment)
        axis.fill_between(sizes, q05, q95, color="#56B4E9", alpha=0.25, linewidth=0)
        axis.fill_between(sizes, q25, q75, color="#0072B2", alpha=0.32, linewidth=0)
        axis.plot(sizes, q50, color="#0072B2", linewidth=1.8)
        axis.axhline(gold_ate, color="0.25", linewidth=1.1, linestyle=":")
        axis.set_title(SCENARIO_LABELS[scenario])
        axis.set_xlabel("Gold-labeled clusters")
        axis.set_xlim(1, clusters)
        style_axis(axis)
    axes[0].set_ylabel("Design-scaled ATE estimate")
    figure.tight_layout(w_pad=1.5)
    figure.savefig(
        Path(cfg["outdir"]) / "fig2_budget_distributions.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def find_budget_row(
    rows: list[dict[str, Any]], scenario: str, size: int
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["scenario"] == scenario and row["validated_clusters"] == size
    )


def write_macros(
    path: Path,
    scenario_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> None:
    size = max(1, cfg["clusters"] // 4)
    concentrated = find_budget_row(budget_rows, "concentrated", size)
    diffuse = find_budget_row(budget_rows, "diffuse", size)
    scenario_lookup = {row["scenario"]: row for row in scenario_rows}
    experiment = make_experiment(
        "concentrated",
        cfg["clusters"],
        cfg["cluster_size"],
        cfg["seed"],
    )
    raw = raw_swap_distributions(experiment)[size]
    proxy_ate = difference_in_means(experiment.proxy, experiment.treatment)
    zero_move = float(np.mean(np.isclose(raw, proxy_ate)))
    commands = {
        "NumClusters": str(cfg["clusters"]),
        "ClusterSize": str(cfg["cluster_size"]),
        "GoldATE": format_number(scenario_lookup["diffuse"]["gold_ate"]),
        "ProxyATE": format_number(scenario_lookup["diffuse"]["proxy_ate"]),
        "EndpointGap": format_number(scenario_lookup["diffuse"]["endpoint_gap"]),
        "PredictionRMSE": format_number(scenario_lookup["diffuse"]["rmse"]),
        "BudgetExample": str(size),
        "ConcentratedZeroMove": format_percent(zero_move, 1),
        "DiffuseBudgetSD": format_number(diffuse["sd"]),
        "ConcentratedBudgetSD": format_number(concentrated["sd"]),
        "ConcentratedTailRisk": format_percent(concentrated["tail_probability"], 1),
        "TailThreshold": format_number(TAIL_THRESHOLD),
    }
    path.write_text(
        "".join(
            f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in commands.items()
        ),
        encoding="utf-8",
    )


def write_metadata(cfg: dict[str, Any]) -> None:
    metadata = {key: value for key, value in cfg.items() if key != "outdir"}
    metadata.update(
        {
            "scenarios": list(SCENARIOS),
            "target_ate": TARGET_ATE,
            "target_rmse": TARGET_RMSE,
            "total_distortion": TOTAL_DISTORTION,
            "tail_threshold": TAIL_THRESHOLD,
            "enumeration": "exact fixed-size subsets",
        }
    )
    (Path(cfg["outdir"]) / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    cfg = configuration(args)
    output_directory = Path(cfg["outdir"])
    output_directory.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print("Validation-swap distributions")
    print(
        f"  clusters={cfg['clusters']}, cluster_size={cfg['cluster_size']}, "
        f"seed={cfg['seed']}"
    )
    experiments = make_experiments(cfg)
    scenario_rows = scenario_table(cfg, experiments)
    budget_rows = budget_table(cfg, experiments)
    figure_same_mean(cfg, experiments)
    figure_budget_distributions(cfg, experiments)
    latex_scenario_table(output_directory / "table_scenarios.tex", scenario_rows)
    latex_budget_table(
        output_directory / "table_budgets.tex", budget_rows, cfg["clusters"]
    )
    write_macros(output_directory / "macros.tex", scenario_rows, budget_rows, cfg)
    write_metadata(cfg)
    elapsed = time.time() - started
    print(f"Completed in {elapsed:.1f}s. Outputs: {output_directory}")


if __name__ == "__main__":
    main()
