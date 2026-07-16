"""Experiment 5 -- Dual-Theater Case Studies with Sensitivity Analysis.

Part A -- Cross-theater comparison:
  Five baselines (Greedy, EV, SAA, Minimax, DRSO) on Indo-Pacific and European
  theaters. Metrics: worst-case SWR, decision latency.

Part B -- Sensitivity analysis:
  B1: Wasserstein radius sweep (both theaters).
  B2: +/-30% scenario weight perturbation -> assignment stability (DRSO vs EV).

Part C -- A2/AD radius sweep (Indo-Pacific only):
  Radii [300, 500, 600, 900, 1500] km from threat center (30N, 130E).
  Trace DRSO posture directives as contested zone expands.

Outputs
-------
results/figures/exp5_theater_comparison.pdf / .png
results/figures/exp5_sensitivity.pdf / .png
results/figures/exp5_a2ad.pdf / .png
results/tables/exp5_theater_comparison.tex
results/tables/exp5_sensitivity.tex
results/tables/exp5_a2ad.tex
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from postureopt.core import (
    Asset, AssetType, PostureOptimizer, ScenarioWeightedOptimizer, ThreatScenario,
)
from postureopt.data import (
    A2AD_THREAT_CENTER,
    _INDOPACIFIC_LOCATIONS,
    _EUROPEAN_LOCATIONS,
    make_indopacific_theater,
    make_european_theater,
    make_indopacific_scenarios,
    make_european_scenarios,
    make_a2ad_scenarios,
    make_robustness_scenarios,
)
from postureopt.drsp import AdversarialModel, MinimaxOptimizer, RobustCEVOptimizer
from postureopt.evaluate import (
    assignment_stability,
    contested_zone_coverage,
    scenario_weighted_readiness,
)
from postureopt.stats import confidence_interval, paired_ttest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SEEDS       = 10
N_SC_TRAIN    = 20   # scenarios for optimizer
N_SC_OOS      = 50   # out-of-sample evaluation scenarios
EPSILONS_SENS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
A2AD_RADII    = [300, 500, 600, 900, 1500]   # km

RESULTS = Path(__file__).parent.parent / "results"
FIG_DIR = RESULTS / "figures"
TAB_DIR = RESULTS / "tables"

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
GREEN  = "#1a9641"
PURPLE = "#762a83"
GRAY   = "#636363"
COLORS = [GRAY, BLUE, GREEN, ORANGE, PURPLE]
BASELINES = ["Greedy", "EV", "SAA", "Minimax", "DRSO"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply(assets, assignment):
    return [
        Asset(a.asset_id, a.asset_type,
              assignment.get(a.asset_id, a.location_id),
              a.quantity, a.readiness_rate, a.maintenance_days_remaining)
        for a in assets
    ]


def _run_all_baselines(assets, locs, train_scenarios, oos_scenarios, seed):
    """Return dict: baseline -> (worst_case_swr, latency_ms).

    EV  = single scenario at mean threat level (deterministic expected-value).
    SAA = all training scenarios, uniform weights (sample average approximation).
    """
    adversary = AdversarialModel(p_obs=0.7, rationality=1.0)

    # EV: single mean-threat scenario — the true deterministic baseline
    loc_ids = list(train_scenarios[0].threat_levels.keys()) if train_scenarios else []
    mean_threats = {
        lid: float(np.mean([s.threat_levels.get(lid, 0.0) for s in train_scenarios]))
        for lid in loc_ids
    }
    ev_scenarios = [ThreatScenario("EV_mean", mean_threats, weight=1.0)]
    saa_scenarios = train_scenarios  # SAA uses full scenario set

    results = {}

    # Greedy
    t0 = time.perf_counter()
    g_assign = PostureOptimizer(seed=seed).greedy_placement(assets, locs)
    lat_g = (time.perf_counter() - t0) * 1000
    results["Greedy"] = (
        scenario_weighted_readiness(_apply(assets, g_assign), oos_scenarios), lat_g
    )

    # EV (single mean-threat scenario)
    t0 = time.perf_counter()
    ev_assign = ScenarioWeightedOptimizer(seed=seed).optimize_placement(
        assets, locs, ev_scenarios
    )
    lat_ev = (time.perf_counter() - t0) * 1000
    results["EV"] = (
        scenario_weighted_readiness(_apply(assets, ev_assign), oos_scenarios), lat_ev
    )

    # SAA (ScenarioWeightedOptimizer with full scenario set)
    t0 = time.perf_counter()
    saa_assign = ScenarioWeightedOptimizer(seed=seed).optimize_placement(
        assets, locs, saa_scenarios
    )
    lat_saa = (time.perf_counter() - t0) * 1000
    results["SAA"] = (
        scenario_weighted_readiness(_apply(assets, saa_assign), oos_scenarios), lat_saa
    )

    # Minimax
    t0 = time.perf_counter()
    mm_assign = MinimaxOptimizer(seed=seed).optimize_placement(
        assets, locs, train_scenarios
    )
    lat_mm = (time.perf_counter() - t0) * 1000
    results["Minimax"] = (
        scenario_weighted_readiness(_apply(assets, mm_assign), oos_scenarios), lat_mm
    )

    # DRSO
    t0 = time.perf_counter()
    drso_assign, _ = RobustCEVOptimizer(adversary, seed=seed).optimize_placement(
        assets, locs, train_scenarios
    )
    lat_drso = (time.perf_counter() - t0) * 1000
    results["DRSO"] = (
        scenario_weighted_readiness(_apply(assets, drso_assign), oos_scenarios), lat_drso
    )

    return results


def _oos_scenarios_for(theater: str, seed: int):
    """Adversarial out-of-sample scenarios for final evaluation."""
    if theater == "IndoPacific":
        locs = _INDOPACIFIC_LOCATIONS
    else:
        locs = _EUROPEAN_LOCATIONS
    return make_robustness_scenarios(locs, epsilon=0.9, n_scenarios=N_SC_OOS, seed=seed + 1000)


# ---------------------------------------------------------------------------
# Part A -- Cross-theater comparison
# ---------------------------------------------------------------------------

def run_theater_comparison():
    """Returns: {theater: {baseline: {"swr": [...], "lat": [...]}}}"""
    theaters = {
        "IndoPacific": (make_indopacific_theater, make_indopacific_scenarios),
        "European":    (make_european_theater,    make_european_scenarios),
    }
    all_results = {}
    for name, (make_theater, make_scenarios) in theaters.items():
        print(f"  Theater: {name}")
        seed_data: Dict[str, Dict[str, list]] = {b: {"swr": [], "lat": []} for b in BASELINES}
        for seed in range(N_SEEDS):
            state = make_theater()
            train = make_scenarios(n_scenarios=N_SC_TRAIN, seed=seed)
            oos   = _oos_scenarios_for(name, seed)
            res   = _run_all_baselines(state.assets, state.locations, train, oos, seed)
            for b in BASELINES:
                swr, lat = res[b]
                seed_data[b]["swr"].append(swr)
                seed_data[b]["lat"].append(lat)
            print(f"    seed {seed}: DRSO={res['DRSO'][0]:.4f}  Greedy={res['Greedy'][0]:.4f}")
        all_results[name] = seed_data
    return all_results


# ---------------------------------------------------------------------------
# Part B -- Sensitivity analysis
# ---------------------------------------------------------------------------

def run_epsilon_sweep():
    """Worst-case SWR vs epsilon for EV, SAA, DRSO in each theater."""
    theaters = {
        "IndoPacific": (make_indopacific_theater, _INDOPACIFIC_LOCATIONS),
        "European":    (make_european_theater,    _EUROPEAN_LOCATIONS),
    }
    results = {}
    for name, (make_theater, locs) in theaters.items():
        results[name] = {b: [] for b in ["EV", "SAA", "DRSO"]}
        for eps in EPSILONS_SENS:
            swr_ev, swr_saa, swr_drso = [], [], []
            for seed in range(N_SEEDS):
                state = make_theater()
                train = make_robustness_scenarios(locs, epsilon=eps,
                                                  n_scenarios=N_SC_TRAIN, seed=seed)
                oos   = make_robustness_scenarios(locs, epsilon=0.9,
                                                  n_scenarios=N_SC_OOS, seed=seed + 1000)
                ev_a = ScenarioWeightedOptimizer(seed=seed).optimize_placement(
                    state.assets, locs, train)
                saa_a = ScenarioWeightedOptimizer(seed=seed).optimize_placement(
                    state.assets, locs, train)
                drso_a, _ = RobustCEVOptimizer(
                    AdversarialModel(0.7, 1.0), seed=seed
                ).optimize_placement(state.assets, locs, train)
                swr_ev.append(scenario_weighted_readiness(_apply(state.assets, ev_a), oos))
                swr_saa.append(scenario_weighted_readiness(_apply(state.assets, saa_a), oos))
                swr_drso.append(scenario_weighted_readiness(_apply(state.assets, drso_a), oos))
            results[name]["EV"].append(float(np.mean(swr_ev)))
            results[name]["SAA"].append(float(np.mean(swr_saa)))
            results[name]["DRSO"].append(float(np.mean(swr_drso)))
    return results


def run_perturbation_stability():
    """Assignment stability (DRSO vs EV) under +-30% weight perturbation."""
    import random
    theaters = {
        "IndoPacific": (make_indopacific_theater, make_indopacific_scenarios),
        "European":    (make_european_theater,    make_european_scenarios),
    }
    results = {}
    for name, (make_theater, make_scenarios) in theaters.items():
        drso_stabs, ev_stabs = [], []
        for seed in range(N_SEEDS):
            state = make_theater()
            base = make_scenarios(n_scenarios=N_SC_TRAIN, seed=seed)
            rng = random.Random(seed + 500)
            perturbed = [
                ThreatScenario(s.scenario_id, s.threat_levels,
                               weight=s.weight * rng.uniform(0.70, 1.30))
                for s in base
            ]
            locs = state.locations
            ev_base = ScenarioWeightedOptimizer(seed=seed).optimize_placement(
                state.assets, locs, base)
            ev_pert = ScenarioWeightedOptimizer(seed=seed).optimize_placement(
                state.assets, locs, perturbed)
            drso_base, _ = RobustCEVOptimizer(
                AdversarialModel(0.7, 1.0), seed=seed
            ).optimize_placement(state.assets, locs, base)
            drso_pert, _ = RobustCEVOptimizer(
                AdversarialModel(0.7, 1.0), seed=seed
            ).optimize_placement(state.assets, locs, perturbed)
            drso_stabs.append(assignment_stability(drso_base, drso_pert))
            ev_stabs.append(assignment_stability(ev_base, ev_pert))
        results[name] = {
            "drso_mean": float(np.mean(drso_stabs)),
            "ev_mean":   float(np.mean(ev_stabs)),
            "drso_stabs": drso_stabs,
            "ev_stabs":   ev_stabs,
        }
    return results


# ---------------------------------------------------------------------------
# Part C -- A2/AD radius sweep
# ---------------------------------------------------------------------------

def run_a2ad_sweep():
    """Track posture changes as A2/AD radius expands over Indo-Pacific.

    Training scenarios reflect the current A2/AD radius (contested zone).
    OOS evaluation uses a fixed adversarial theater scenario set so that
    SWR is comparable across radii and shows true placement degradation.
    """
    from postureopt.core import haversine_distance
    lat, lon = A2AD_THREAT_CENTER
    locs = _INDOPACIFIC_LOCATIONS

    rows = []
    for radius in A2AD_RADII:
        swr_by_baseline: Dict[str, list] = {b: [] for b in BASELINES}
        cov_drso: list = []
        cov_greedy: list = []
        cov_ev: list = []
        for seed in range(N_SEEDS):
            state = make_indopacific_theater()
            # Training: A2/AD scenarios at this radius (defender sees current threat)
            train = make_a2ad_scenarios(locs, lat, lon, radius,
                                        n_scenarios=N_SC_TRAIN, seed=seed)
            # OOS: fixed theater adversarial scenarios (consistent evaluation baseline)
            oos = make_indopacific_scenarios(n_scenarios=N_SC_OOS, seed=seed + 2000)
            res = _run_all_baselines(state.assets, state.locations, train, oos, seed)
            for b in BASELINES:
                swr_by_baseline[b].append(res[b][0])
            # Contested zone coverage: DRSO, Greedy, EV for comparison
            drso_assign, _ = RobustCEVOptimizer(
                AdversarialModel(0.7, 1.0), seed=seed
            ).optimize_placement(state.assets, locs, train)
            greedy_assign = PostureOptimizer(seed=seed).greedy_placement(state.assets, locs)
            loc_ids = list(train[0].threat_levels.keys())
            mean_threats = {
                lid: float(np.mean([s.threat_levels.get(lid, 0.0) for s in train]))
                for lid in loc_ids
            }
            ev_assign = ScenarioWeightedOptimizer(seed=seed).optimize_placement(
                state.assets, locs, [ThreatScenario("EV_mean", mean_threats, weight=1.0)]
            )
            cov_drso.append(contested_zone_coverage(drso_assign, locs, lat, lon, radius))
            cov_greedy.append(contested_zone_coverage(greedy_assign, locs, lat, lon, radius))
            cov_ev.append(contested_zone_coverage(ev_assign, locs, lat, lon, radius))

        # Count locations within radius
        n_contested = sum(
            1 for loc in locs
            if haversine_distance(lat, lon, loc.lat, loc.lon) <= radius
        )
        contested_names = [
            loc.name for loc in locs
            if haversine_distance(lat, lon, loc.lat, loc.lon) <= radius
        ]
        rows.append({
            "radius": radius,
            "n_contested": n_contested,
            "contested_names": contested_names,
            "cov_drso":   float(np.mean(cov_drso)),
            "cov_greedy": float(np.mean(cov_greedy)),
            "cov_ev":     float(np.mean(cov_ev)),
            **{f"swr_{b}": float(np.mean(swr_by_baseline[b])) for b in BASELINES},
        })
        print(f"    radius={radius}km: {n_contested} locs contested | "
              f"DRSO SWR={rows[-1]['swr_DRSO']:.4f} cov={rows[-1]['cov_drso']:.3f} | "
              f"Greedy cov={rows[-1]['cov_greedy']:.3f}")
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_theater_figure(comp_results):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    fig.subplots_adjust(wspace=0.40, left=0.08, right=0.97, bottom=0.18, top=0.88)

    theater_labels = ["IndoPacific", "European"]
    x = np.arange(len(BASELINES))
    width = 0.35

    for ax, t_label, short in zip(axes, theater_labels, ["(a) Indo-Pacific", "(b) European"]):
        means = [np.mean(comp_results[t_label][b]["swr"]) for b in BASELINES]
        errs  = [np.std(comp_results[t_label][b]["swr"]) for b in BASELINES]
        bars = ax.bar(x, means, width=0.6, color=COLORS, alpha=0.80, yerr=errs,
                      capsize=3, error_kw={"elinewidth": 1.0})
        ax.set_xticks(x)
        ax.set_xticklabels(BASELINES, rotation=15, ha="right")
        ax.set_ylabel("Worst-case SWR (out-of-sample)")
        ax.set_title(f"{short} theater")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    return fig


def make_sensitivity_figure(eps_results, stab_results):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    fig.subplots_adjust(wspace=0.40, left=0.09, right=0.97, bottom=0.16, top=0.88)

    # Left: epsilon sweep
    ax = axes[0]
    for theater, ls in [("IndoPacific", "-"), ("European", "--")]:
        for b, col in [("EV", BLUE), ("DRSO", ORANGE)]:
            vals = eps_results[theater][b]
            label = f"{b} ({theater[:4]})"
            ax.plot(EPSILONS_SENS, vals, color=col, ls=ls, marker="o", ms=4, label=label)
    ax.set_xlabel("Wasserstein radius proxy epsilon")
    ax.set_ylabel("Worst-case SWR (out-of-sample)")
    ax.set_title("(a) Epsilon sensitivity (both theaters)")
    ax.legend(fontsize=7, loc="lower right")

    # Right: perturbation stability bars
    ax = axes[1]
    theater_labels = list(stab_results.keys())
    x = np.arange(len(theater_labels))
    w = 0.30
    drso_means = [stab_results[t]["drso_mean"] for t in theater_labels]
    ev_means   = [stab_results[t]["ev_mean"]   for t in theater_labels]
    ax.bar(x - w / 2, drso_means, width=w, color=ORANGE, alpha=0.80, label="DRSO")
    ax.bar(x + w / 2, ev_means,   width=w, color=BLUE,   alpha=0.80, label="EV")
    ax.set_xticks(x)
    ax.set_xticklabels(["Indo-Pacific", "European"])
    ax.set_ylabel("Assignment stability (fraction unchanged)")
    ax.set_title("(b) Stability under +/-30% weight perturbation")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.85, color=ORANGE, ls=":", lw=1.2, alpha=0.7, label="85% threshold")
    ax.legend(fontsize=8)

    return fig


def make_a2ad_figure(a2ad_rows):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    fig.subplots_adjust(wspace=0.40, left=0.09, right=0.97, bottom=0.16, top=0.88)

    radii = [r["radius"] for r in a2ad_rows]

    # Left: SWR vs radius — show all baselines, widen y-axis for readability
    ax = axes[0]
    for b, col in zip(BASELINES, COLORS):
        ax.plot(radii, [r[f"swr_{b}"] for r in a2ad_rows],
                color=col, marker="o", ms=4, label=b)
    all_swr = [r[f"swr_{b}"] for r in a2ad_rows for b in BASELINES]
    ymin, ymax = min(all_swr), max(all_swr)
    pad = max((ymax - ymin) * 0.5, 0.005)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("A2/AD contested zone radius (km)")
    ax.set_ylabel("Worst-case SWR (out-of-sample)")
    ax.set_title("(a) Worst-case readiness vs. A2/AD radius")
    ax.legend(fontsize=8)

    # Right: survivable coverage — DRSO vs Greedy vs EV
    ax = axes[1]
    ax.plot(radii, [r["cov_drso"]   for r in a2ad_rows],
            color=ORANGE, marker="s", ms=5, label="DRSO")
    ax.plot(radii, [r["cov_greedy"] for r in a2ad_rows],
            color=GRAY,   marker="^", ms=5, label="Greedy", ls="--")
    ax.plot(radii, [r["cov_ev"]     for r in a2ad_rows],
            color=BLUE,   marker="o", ms=5, label="EV", ls=":")
    ax.set_xlabel("A2/AD contested zone radius (km)")
    ax.set_ylabel("Survivable coverage (fraction outside zone)")
    ax.set_title("(b) Posture shift as A2/AD matures")
    # Annotate contested location count at each radius transition
    prev = 0
    for r in a2ad_rows:
        if r["n_contested"] != prev:
            ax.axvline(r["radius"], color="black", ls=":", lw=0.8, alpha=0.4)
            ax.annotate(
                f"{r['n_contested']} loc. in zone",
                (r["radius"], 0.08), rotation=90,
                fontsize=7, color="black", alpha=0.7,
                va="bottom", ha="right",
            )
            prev = r["n_contested"]
    ax.set_ylim(-0.05, 1.10)
    ax.legend(fontsize=8)

    return fig


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def latex_theater_table(comp_results):
    rows = []
    for theater in ["IndoPacific", "European"]:
        for b in BASELINES:
            vals = comp_results[theater][b]["swr"]
            lat_vals = comp_results[theater][b]["lat"]
            ci = confidence_interval(vals)
            rows.append(
                f"        {theater} & {b}"
                f" & {ci['mean']:.4f}"
                f" & {ci['lower']:.4f}--{ci['upper']:.4f}"
                f" & {np.mean(lat_vals):.2f} \\\\"
            )
        rows.append(r"        \hline")
    body = "\n".join(rows)
    return (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        r"\caption{Experiment 5A: cross-theater comparison of five baselines. "
        r"Worst-case SWR evaluated on 50 adversarial out-of-sample scenarios. "
        r"95\% CI over 10 seeds. Latency = wall-clock ms from theater state to "
        r"ranked posture directive.}" "\n"
        r"\label{tab:exp5_theater}" "\n"
        r"\begin{tabular}{llccc}" "\n"
        r"\hline" "\n"
        r"Theater & Baseline & Mean SWR & 95\% CI & Latency (ms) \\" "\n"
        r"\hline" "\n"
        f"{body}\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


def latex_sensitivity_table(eps_results, stab_results):
    rows = []
    for theater in ["IndoPacific", "European"]:
        for eps, ev_v, drso_v in zip(
            EPSILONS_SENS,
            eps_results[theater]["EV"],
            eps_results[theater]["DRSO"],
        ):
            rows.append(
                f"        {theater} & {eps:.2f}"
                f" & {ev_v:.4f} & {drso_v:.4f} \\\\"
            )
        rows.append(r"        \hline")
    rows.append(r"        \multicolumn{4}{l}{\textit{Assignment stability under $\pm30\%$ weight perturbation}} \\")
    rows.append(r"        \hline")
    for theater in ["IndoPacific", "European"]:
        s = stab_results[theater]
        rows.append(
            f"        {theater} & -- & EV: {s['ev_mean']:.3f} & DRSO: {s['drso_mean']:.3f} \\\\"
        )
    body = "\n".join(rows)
    return (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        r"\caption{Experiment 5B: epsilon sensitivity and assignment stability. "
        r"Worst-case SWR averaged over 10 seeds. Stability = fraction of assets "
        r"with unchanged location under $\pm30\%$ scenario weight perturbation.}" "\n"
        r"\label{tab:exp5_sensitivity}" "\n"
        r"\begin{tabular}{llcc}" "\n"
        r"\hline" "\n"
        r"Theater & $\varepsilon$ & EV SWR & DRSO SWR \\" "\n"
        r"\hline" "\n"
        f"{body}\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


def latex_a2ad_table(a2ad_rows):
    body = "\n".join(
        f"        {r['radius']}"
        f" & {r['n_contested']}"
        f" & {r['swr_DRSO']:.4f}"
        f" & {r['swr_EV']:.4f}"
        f" & {r['cov_drso']:.3f}"
        f" & {r['cov_greedy']:.3f}"
        f" & {r['cov_ev']:.3f} \\\\"
        for r in a2ad_rows
    )
    return (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        r"\caption{Experiment 5C: A2/AD contested zone radius sweep (Indo-Pacific). "
        r"Threat center at (30\textdegree N, 130\textdegree E). "
        r"SWR evaluated against fixed theater adversarial scenarios. "
        r"Coverage = fraction of assets placed outside contested zone "
        r"(higher is more survivable).}" "\n"
        r"\label{tab:exp5_a2ad}" "\n"
        r"\begin{tabular}{cccccccc}" "\n"
        r"\hline" "\n"
        r"Radius (km) & Locs. in zone & DRSO SWR & EV SWR & DRSO cov. & Greedy cov. & EV cov. \\" "\n"
        r"\hline" "\n"
        f"{body}\n"
        r"\hline" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(comp_results, eps_results, stab_results, a2ad_rows):
    W = 76
    print(f"\n{'='*W}")
    print("  EXPERIMENT 5A -- CROSS-THEATER BASELINE COMPARISON")
    print(f"{'='*W}")
    for theater in ["IndoPacific", "European"]:
        print(f"\n  Theater: {theater}")
        print(f"  {'Baseline':>8}  {'Mean SWR':>10}  {'Std':>7}  {'Latency ms':>11}")
        print("  " + "-" * 44)
        for b in BASELINES:
            vals = comp_results[theater][b]["swr"]
            lats = comp_results[theater][b]["lat"]
            print(f"  {b:>8}  {np.mean(vals):>10.4f}  {np.std(vals):>7.4f}  {np.mean(lats):>11.3f}")

    print(f"\n{'='*W}")
    print("  EXPERIMENT 5B -- SENSITIVITY ANALYSIS")
    print(f"{'='*W}")
    for theater in ["IndoPacific", "European"]:
        s = stab_results[theater]
        ttest = paired_ttest(s["drso_stabs"], s["ev_stabs"])
        print(f"\n  {theater}: DRSO stability={s['drso_mean']:.3f}  "
              f"EV stability={s['ev_mean']:.3f}  "
              f"p={ttest['p_value']:.4f}  "
              f"reject_null={ttest['reject_null']}")

    print(f"\n{'='*W}")
    print("  EXPERIMENT 5C -- A2/AD RADIUS SWEEP (Indo-Pacific)")
    print(f"{'='*W}")
    print(f"  {'Radius':>8}  {'Contested':>10}  {'DRSO SWR':>10}  {'EV SWR':>8}  {'Coverage':>9}")
    print("  " + "-" * 52)
    for r in a2ad_rows:
        print(f"  {r['radius']:>8}  {r['n_contested']:>10}  {r['swr_DRSO']:>10.4f}"
              f"  {r['swr_EV']:>8.4f}  {r['cov_drso']:>9.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Running Experiment 5A -- cross-theater ({N_SEEDS} seeds) ...")
    comp_results = run_theater_comparison()

    print(f"\nRunning Experiment 5B -- epsilon sweep ...")
    eps_results = run_epsilon_sweep()

    print(f"\nRunning Experiment 5B -- perturbation stability ...")
    stab_results = run_perturbation_stability()

    print(f"\nRunning Experiment 5C -- A2/AD radius sweep ...")
    a2ad_rows = run_a2ad_sweep()

    print_summary(comp_results, eps_results, stab_results, a2ad_rows)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    for fig, stem in [
        (make_theater_figure(comp_results),                "exp5_theater_comparison"),
        (make_sensitivity_figure(eps_results, stab_results), "exp5_sensitivity"),
        (make_a2ad_figure(a2ad_rows),                      "exp5_a2ad"),
    ]:
        for ext in ("pdf", "png"):
            p = FIG_DIR / f"{stem}.{ext}"
            fig.savefig(p, bbox_inches="tight")
            print(f"\n  Figure -> {p}")
        plt.close(fig)

    tables = {
        "exp5_theater_comparison.tex": latex_theater_table(comp_results),
        "exp5_sensitivity.tex":        latex_sensitivity_table(eps_results, stab_results),
        "exp5_a2ad.tex":               latex_a2ad_table(a2ad_rows),
    }
    for fname, content in tables.items():
        p = TAB_DIR / fname
        p.write_text(content)
        print(f"  Table  -> {p}")
        print(f"\n{content}")


if __name__ == "__main__":
    main()
