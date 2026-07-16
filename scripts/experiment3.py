"""Experiment 3 — Adversarial Robustness: Bayesian Counter-Move Model.

Setup : 20 assets, 8 Indo-Pacific locations.
Vary  : observation probability p_obs ∈ {0, 0.25, 0.5, 0.75, 1.0}
        adversary rationality ∈ {random (0.0), Bayesian (1.0)}
Compare: naive CEV, robust CEV, greedy — all evaluated under the adversarially-
         updated scenario distribution that each placement triggers.
Metrics: scenario-weighted readiness (SWR) and adversarial regret
         (robust SWR − naive SWR).

Outputs
-------
results/figures/exp3_main.pdf / .png  — 2-panel figure
results/tables/exp3_results.tex       — full results table (LaTeX)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from postureopt.core import (
    Asset, AssetType, Location, PostureOptimizer,
    ScenarioWeightedOptimizer, ThreatScenario,
)
from postureopt.data import make_robustness_scenarios
from postureopt.drsp import AdversarialModel, RobustCEVOptimizer
from postureopt.evaluate import scenario_weighted_readiness

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_ASSETS    = 20
N_SCENARIOS = 20
N_SEEDS     = 10
P_OBS_VALS  = [0.0, 0.25, 0.50, 0.75, 1.0]
RATIONALITY = {"Bayesian": 1.0, "Random": 0.0}

RESULTS = Path(__file__).parent.parent / "results"
FIG_DIR = RESULTS / "figures"
TAB_DIR = RESULTS / "tables"

INDOPACIFIC_LOCS = [
    Location("L1", "Kadena AB",      26.35, 127.77, capacity=10, strategic_value=0.95),
    Location("L2", "Andersen AFB",   13.58, 144.93, capacity=10, strategic_value=0.90),
    Location("L3", "MCAS Iwakuni",   34.14, 132.24, capacity=10, strategic_value=0.85),
    Location("L4", "Camp HM Smith",  21.41,-157.93, capacity=10, strategic_value=0.80),
    Location("L5", "Diego Garcia",    -7.32,  72.42, capacity=10, strategic_value=0.78),
    Location("L6", "Misawa AB",      40.70, 141.37, capacity=10, strategic_value=0.75),
    Location("L7", "Osan AB",        37.09, 127.03, capacity=10, strategic_value=0.88),
    Location("L8", "Clark AB",       15.19, 120.56, capacity=10, strategic_value=0.72),
]

mpl.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.8, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linestyle": "--",
})

BLUE   = "#2166ac"
ORANGE = "#d6604d"
GRAY   = "#636363"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assets(seed: int) -> List[Asset]:
    import random
    rng = random.Random(seed)
    return [
        Asset(f"A{i:03d}", AssetType.AIRCRAFT,
              INDOPACIFIC_LOCS[i % len(INDOPACIFIC_LOCS)].location_id,
              quantity=2, readiness_rate=round(rng.uniform(0.6, 1.0), 3))
        for i in range(N_ASSETS)
    ]


def _apply(assets: List[Asset], assignment: Dict[str, str]) -> List[Asset]:
    return [
        Asset(a.asset_id, a.asset_type,
              assignment.get(a.asset_id, a.location_id),
              a.quantity, a.readiness_rate, a.maintenance_days_remaining)
        for a in assets
    ]


def _prior_scenarios(seed: int) -> List[ThreatScenario]:
    """90% safe, 10% concentrated on highest-value locations."""
    return make_robustness_scenarios(
        INDOPACIFIC_LOCS, epsilon=0.1, n_scenarios=N_SCENARIOS, seed=seed
    )


# ---------------------------------------------------------------------------
# Single-condition run
# ---------------------------------------------------------------------------


def run_condition(p_obs: float, rationality: float, seed: int) -> Dict[str, float]:
    """Return SWR for greedy, naive CEV, and robust CEV under p_obs / rationality."""
    assets   = _make_assets(seed)
    locs     = INDOPACIFIC_LOCS
    prior    = _prior_scenarios(seed)
    adversary = AdversarialModel(p_obs=p_obs, rationality=rationality)

    # Greedy
    greedy_assignment = PostureOptimizer(seed=seed).greedy_placement(assets, locs)
    adv_for_greedy    = adversary.update_weights(prior, greedy_assignment, locs)
    greedy_swr        = scenario_weighted_readiness(_apply(assets, greedy_assignment), adv_for_greedy)

    # Naive CEV (optimises against prior, unaware of adversarial response)
    naive_assignment = ScenarioWeightedOptimizer(seed=seed).optimize_placement(assets, locs, prior)
    adv_for_naive    = adversary.update_weights(prior, naive_assignment, locs)
    naive_swr        = scenario_weighted_readiness(_apply(assets, naive_assignment), adv_for_naive)

    # Robust CEV (adversary-aware iteration)
    robust = RobustCEVOptimizer(adversary, seed=seed)
    robust_assignment, adv_for_robust = robust.optimize_placement(assets, locs, prior)
    robust_swr = scenario_weighted_readiness(_apply(assets, robust_assignment), adv_for_robust)

    return {
        "greedy": greedy_swr,
        "naive_cev": naive_swr,
        "robust_cev": robust_swr,
        "adv_regret": robust_swr - naive_swr,
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep() -> Dict[str, Dict]:
    """results[rationality_label][p_obs] = mean metrics over N_SEEDS."""
    results = {}
    for rat_label, rat_val in RATIONALITY.items():
        results[rat_label] = {}
        for p in P_OBS_VALS:
            seed_results = [run_condition(p, rat_val, seed) for seed in range(N_SEEDS)]
            results[rat_label][p] = {
                k: float(np.mean([r[k] for r in seed_results]))
                for k in seed_results[0]
            }
    return results


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(results: Dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
    fig.subplots_adjust(wspace=0.38, left=0.08, right=0.97, bottom=0.16, top=0.88)

    for ax, rat_label in zip(axes, ["Bayesian", "Random"]):
        r = results[rat_label]
        ps    = P_OBS_VALS
        gr    = [r[p]["greedy"]     for p in ps]
        naive = [r[p]["naive_cev"]  for p in ps]
        rob   = [r[p]["robust_cev"] for p in ps]

        ax.plot(ps, gr,    color=GRAY,   ls=":",  label="Greedy")
        ax.plot(ps, naive, color=ORANGE, ls="--", label="Naive CEV")
        ax.plot(ps, rob,   color=BLUE,   ls="-",  label="Robust CEV")
        ax.set_xlabel("Adversary observation probability $p_{obs}$")
        ax.set_ylabel("Scenario-Weighted Readiness")
        ax.set_title(f"({'a' if rat_label == 'Bayesian' else 'b'}) "
                     f"{rat_label} adversary")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks(ps)
        ax.legend(loc="lower left", framealpha=0.85)

    return fig


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------


def latex_table(results: Dict) -> str:
    rows = []
    for rat_label in ["Bayesian", "Random"]:
        for p in P_OBS_VALS:
            r = results[rat_label][p]
            rows.append(
                f"        {rat_label} & {p:.2f}"
                f" & {r['greedy']:.4f}"
                f" & {r['naive_cev']:.4f}"
                f" & {r['robust_cev']:.4f}"
                f" & {r['adv_regret']:+.4f} \\\\"
            )
        rows.append(r"        \hline")

    body = "\n".join(rows)
    return (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        rf"\caption{{Experiment 3: scenario-weighted readiness (SWR) and adversarial regret "
        rf"(robust SWR $-$ naive SWR) across observation probability $p_{{obs}}$ and adversary "
        rf"rationality. Results averaged over {N_SEEDS} seeds; {N_ASSETS} assets, "
        rf"{len(INDOPACIFIC_LOCS)} Indo-Pacific locations, {N_SCENARIOS} scenarios.}}" "\n"
        r"\label{tab:exp3_results}" "\n"
        r"\begin{tabular}{llcccc}" "\n"
        r"\hline" "\n"
        r"Rationality & $p_{obs}$ & Greedy SWR & Naive CEV SWR & Robust CEV SWR & Adv. Regret \\" "\n"
        r"\hline" "\n"
        f"{body}\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def print_summary(results: Dict) -> None:
    W = 76
    print(f"\n{'='*W}")
    print("  EXPERIMENT 3 — ADVERSARIAL ROBUSTNESS")
    print(f"  {N_ASSETS} assets | {len(INDOPACIFIC_LOCS)} locations | "
          f"{N_SCENARIOS} scenarios | {N_SEEDS} seeds")
    print(f"{'='*W}")

    for rat_label in ["Bayesian", "Random"]:
        print(f"\n  Adversary rationality: {rat_label}")
        print(f"  {'p_obs':>6}  {'Greedy':>8}  {'Naive CEV':>10}  "
              f"{'Robust CEV':>11}  {'Adv. Regret':>12}")
        print("  " + "-" * 54)
        for p in P_OBS_VALS:
            r = results[rat_label][p]
            print(f"  {p:>6.2f}  {r['greedy']:>8.4f}  {r['naive_cev']:>10.4f}  "
                  f"{r['robust_cev']:>11.4f}  {r['adv_regret']:>+12.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Running Experiment 3  ({len(P_OBS_VALS)} p_obs values × "
          f"{len(RATIONALITY)} rationality modes × {N_SEEDS} seeds) ...")
    results = run_sweep()
    print_summary(results)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    fig = make_figure(results)
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"exp3_main.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"\n  Figure  → {p}")
    plt.close(fig)

    table = latex_table(results)
    tab_path = TAB_DIR / "exp3_results.tex"
    tab_path.write_text(table)
    print(f"  Table   → {tab_path}")
    print(f"\n{table}")


if __name__ == "__main__":
    main()
