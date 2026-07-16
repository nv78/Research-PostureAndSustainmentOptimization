"""Experiment 1 — Greedy Baseline Characterization (paper-quality output).

Setup  : 20 assets, 5 locations (capacity 5 each), 10 time steps, δ = 0.08/step.
Seeds  : 10 simulation seeds (inner) × 5 scenario seeds (outer).
Metrics: readiness score, coverage score, sustainment cost, posture efficiency.
Threat : greedy SWR under uniform vs. skewed distributions.
Stats  : 95% CIs (t-distribution), paired t-tests with Bonferroni correction,
         and two-level variance decomposition (scenario-seed vs. sim-seed).

Outputs
-------
results/figures/exp1_main.pdf         — 3-panel figure (paper-ready)
results/figures/exp1_main.png         — rasterized copy
results/tables/exp1_metrics.tex       — per-step metrics with 95% CIs (LaTeX)
results/tables/exp1_sensitivity.tex   — threat-sensitivity with 95% CIs (LaTeX)
results/tables/exp1_significance.tex  — paired t-test results table (LaTeX)
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from postureopt.core import (
    Asset,
    Location,
    PostureOptimizer,
    PostureState,
    ReplenishmentPolicy,
)
from postureopt.data import make_posture_state, make_threat_scenarios, simulate_degradation
from postureopt.evaluate import (
    coverage_score,
    posture_efficiency,
    readiness_score,
    scenario_weighted_readiness,
)
from postureopt.stats import (
    bonferroni_correct,
    bootstrap_ci_paired,
    ci_latex,
    cohens_d_paired,
    confidence_interval,
    paired_ttest,
    shapiro_wilk_paired,
    statistical_power_paired,
    variance_decomposition,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_ASSETS = 20
N_LOCATIONS = 5
LOCATION_CAPACITY = 5
N_STEPS = 10
DEGRADATION_RATE = 0.08
SEEDS = list(range(10))                          # inner: simulation seeds
SCENARIO_SEEDS = [999, 1000, 1001, 1002, 1003]  # outer: scenario-set seeds
SCENARIO_SEED = SCENARIO_SEEDS[0]               # primary seed for main tables/figures
N_THREAT_SCENARIOS = 20
CONFIDENCE = 0.95
N_COMPARISONS = 6   # readiness, coverage, cost, efficiency, SWR-uniform, SWR-skewed

RESULTS = Path(__file__).parent.parent / "results"
FIG_DIR = RESULTS / "figures"
TAB_DIR = RESULTS / "tables"

# ---------------------------------------------------------------------------
# Figure style
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

BLUE   = "#2166ac"
GRAY   = "#636363"
ORANGE = "#d6604d"

# ---------------------------------------------------------------------------
# State and placement helpers
# ---------------------------------------------------------------------------


def make_experiment_state(seed: int) -> PostureState:
    base = make_posture_state(n_assets=N_ASSETS, seed=seed)
    locations = [
        Location(
            location_id=loc.location_id, name=loc.name,
            lat=loc.lat, lon=loc.lon,
            capacity=LOCATION_CAPACITY,
            strategic_value=loc.strategic_value,
        )
        for loc in base.locations
    ]
    return PostureState(assets=base.assets, locations=locations, time_step=0)


def apply_assignment(state: PostureState, assignment: Dict[str, str]) -> PostureState:
    new_assets = [
        Asset(
            asset_id=a.asset_id, asset_type=a.asset_type,
            location_id=assignment.get(a.asset_id, a.location_id),
            quantity=a.quantity, readiness_rate=a.readiness_rate,
            maintenance_days_remaining=a.maintenance_days_remaining,
        )
        for a in state.assets
    ]
    return PostureState(assets=new_assets, locations=state.locations, time_step=0)


def random_placement(
    assets: List[Asset], locations: List[Location], rng: random.Random
) -> Dict[str, str]:
    capacity = {loc.location_id: loc.capacity for loc in locations}
    locs = list(locations)
    assignment: Dict[str, str] = {}
    for asset in assets:
        candidates = [l for l in locs if capacity[l.location_id] > 0]
        if candidates:
            chosen = rng.choice(candidates)
            assignment[asset.asset_id] = chosen.location_id
            capacity[chosen.location_id] -= 1
    return assignment


# ---------------------------------------------------------------------------
# Per-step sustainment cost
# ---------------------------------------------------------------------------

ACTION_COSTS = {"REPOSITION": 10.0, "RESUPPLY": 5.0, "MAINTAIN": 2.0, "HOLD": 0.0}


def step_cost(step_state: PostureState) -> float:
    policy = ReplenishmentPolicy()
    return sum(ACTION_COSTS.get(policy.decide(a).value, 0.0) for a in step_state.assets)


# ---------------------------------------------------------------------------
# Single-seed simulation
# ---------------------------------------------------------------------------


def run_seed(
    seed: int, use_greedy: bool
) -> Tuple[PostureState, List[PostureState], Dict[str, str]]:
    state = make_experiment_state(seed)
    opt = PostureOptimizer(seed=seed)
    if use_greedy:
        assignment = opt.greedy_placement(state.assets, state.locations)
    else:
        assignment = random_placement(
            state.assets, state.locations, random.Random(seed + 5000)
        )
    placed = apply_assignment(state, assignment)
    history = simulate_degradation(
        placed, n_steps=N_STEPS, seed=seed, degradation_rate=DEGRADATION_RATE
    )
    return placed, history, assignment


# ---------------------------------------------------------------------------
# Multi-seed metric collection
# ---------------------------------------------------------------------------


def collect_metrics(seeds: List[int], use_greedy: bool) -> Dict[str, np.ndarray]:
    """Arrays shape (n_seeds, N_STEPS+1); index 0 = initial state (t=0)."""
    n = len(seeds)
    readiness  = np.zeros((n, N_STEPS + 1))
    coverage   = np.zeros((n, N_STEPS + 1))
    cost       = np.zeros((n, N_STEPS + 1))
    efficiency = np.zeros((n, N_STEPS + 1))

    for i, seed in enumerate(seeds):
        placed, history, _ = run_seed(seed, use_greedy=use_greedy)
        for t, st in enumerate([placed] + history):
            r = readiness_score(st.assets)
            c = coverage_score(st.assets, st.locations)
            k = step_cost(st)
            readiness[i, t]  = r
            coverage[i, t]   = c
            cost[i, t]       = k
            efficiency[i, t] = posture_efficiency(r, c, k)

    return {"readiness": readiness, "coverage": coverage,
            "cost": cost, "efficiency": efficiency}


def collect_threat_sensitivity(seeds: List[int]) -> Dict[str, np.ndarray]:
    """SWR under uniform and skewed threat at each time step (greedy only)."""
    n = len(seeds)
    swr_uniform = np.zeros((n, N_STEPS + 1))
    swr_skewed  = np.zeros((n, N_STEPS + 1))

    for i, seed in enumerate(seeds):
        placed, history, _ = run_seed(seed, use_greedy=True)
        locs = placed.locations
        sc_u = make_threat_scenarios(locs, distribution="uniform",
                                     n_scenarios=N_THREAT_SCENARIOS,
                                     weight_distribution="uniform", seed=SCENARIO_SEED)
        sc_s = make_threat_scenarios(locs, distribution="skewed",
                                     n_scenarios=N_THREAT_SCENARIOS,
                                     weight_distribution="uniform", seed=SCENARIO_SEED)
        for t, st in enumerate([placed] + history):
            swr_uniform[i, t] = scenario_weighted_readiness(st.assets, sc_u)
            swr_skewed[i, t]  = scenario_weighted_readiness(st.assets, sc_s)

    return {"uniform": swr_uniform, "skewed": swr_skewed}


def collect_variance_decomposition(
    sim_seeds: List[int],
    scenario_seeds: List[int],
    t: int = N_STEPS,
) -> Dict[str, dict]:
    """Two-level variance decomposition of SWR at time step t.

    Outer level: scenario seeds (5) — captures sensitivity to choice of scenario set.
    Inner level: simulation seeds (10) — captures sensitivity to initial-state randomness.
    Returns decompositions for both uniform and skewed threat conditions.
    """
    unif = np.zeros((len(scenario_seeds), len(sim_seeds)))
    skew = np.zeros((len(scenario_seeds), len(sim_seeds)))

    for i, sc_seed in enumerate(scenario_seeds):
        for j, sim_seed in enumerate(sim_seeds):
            placed, history, _ = run_seed(sim_seed, use_greedy=True)
            states = [placed] + history
            locs = placed.locations
            sc_u = make_threat_scenarios(locs, distribution="uniform",
                                         n_scenarios=N_THREAT_SCENARIOS,
                                         weight_distribution="uniform", seed=sc_seed)
            sc_s = make_threat_scenarios(locs, distribution="skewed",
                                         n_scenarios=N_THREAT_SCENARIOS,
                                         weight_distribution="uniform", seed=sc_seed)
            unif[i, j] = scenario_weighted_readiness(states[t].assets, sc_u)
            skew[i, j] = scenario_weighted_readiness(states[t].assets, sc_s)

    return {
        "uniform": variance_decomposition(unif),
        "skewed":  variance_decomposition(skew),
    }


def compute_significance(
    gm: Dict[str, np.ndarray],
    rm: Dict[str, np.ndarray],
    threat: Dict[str, np.ndarray],
) -> Dict[str, dict]:
    """Paired t-tests at t=10 with Cohen's d, bootstrap CIs, Shapiro-Wilk, and power."""
    adj_alpha = bonferroni_correct(1 - CONFIDENCE, N_COMPARISONS)
    t = N_STEPS
    pairs = {
        "readiness":   (gm["readiness"][:, t],  rm["readiness"][:, t]),
        "coverage":    (gm["coverage"][:, t],   rm["coverage"][:, t]),
        "cost":        (gm["cost"][:, t],        rm["cost"][:, t]),
        "efficiency":  (gm["efficiency"][:, t], rm["efficiency"][:, t]),
        "swr_uniform": (threat["uniform"][:, t], threat["skewed"][:, t]),
    }
    results = {}
    for key, (a, b) in pairs.items():
        tt = paired_ttest(a, b, adj_alpha)
        d  = cohens_d_paired(a, b)
        bci = bootstrap_ci_paired(a, b, n_bootstrap=5000, confidence=CONFIDENCE, seed=42)
        sw  = shapiro_wilk_paired(a, b)
        pwr = statistical_power_paired(d, n=len(a), alpha=adj_alpha)
        results[key] = {**tt, "cohens_d": d, "bootstrap_ci": bci,
                        "shapiro_wilk": sw, "power": pwr}
    return results


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(
    gm: Dict[str, np.ndarray],
    rm: Dict[str, np.ndarray],
    threat: Dict[str, np.ndarray],
) -> plt.Figure:
    steps = np.arange(N_STEPS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    fig.subplots_adjust(wspace=0.40, left=0.07, right=0.97, bottom=0.16, top=0.88)

    def _band(ax, data, color, label, ls="-"):
        # Shaded band = ±1σ for visual spread; tables report 95% CIs
        mu = data.mean(axis=0)
        sd = data.std(axis=0, ddof=1)
        ax.plot(steps, mu, color=color, ls=ls, label=label)
        ax.fill_between(steps, mu - sd, mu + sd, color=color, alpha=0.15)

    ax = axes[0]
    _band(ax, gm["readiness"], BLUE, "Greedy")
    _band(ax, rm["readiness"], GRAY, "Random", ls="--")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Readiness Score")
    ax.set_title("(a) Readiness Score")
    ax.set_xlim(0, N_STEPS); ax.set_ylim(0.3, 1.0)
    ax.set_xticks(range(0, N_STEPS + 1, 2))
    ax.legend(loc="lower left", framealpha=0.8)

    ax = axes[1]
    _band(ax, gm["efficiency"], BLUE, "Greedy")
    _band(ax, rm["efficiency"], GRAY, "Random", ls="--")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Posture Efficiency")
    ax.set_title("(b) Posture Efficiency")
    ax.set_xlim(0, N_STEPS)
    ax.set_xticks(range(0, N_STEPS + 1, 2))
    ax.legend(loc="lower left", framealpha=0.8)

    ax = axes[2]
    _band(ax, threat["uniform"], BLUE,   "Uniform threat")
    _band(ax, threat["skewed"],  ORANGE, "Skewed threat")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Scenario-Weighted Readiness")
    ax.set_title("(c) Readiness Under Threat")
    ax.set_xlim(0, N_STEPS); ax.set_ylim(0.0, 0.85)
    ax.set_xticks(range(0, N_STEPS + 1, 2))
    ax.legend(loc="lower left", framealpha=0.8)

    return fig


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------


def _ci_cell(arr: np.ndarray, col: int) -> str:
    return ci_latex(arr[:, col], confidence=CONFIDENCE)


def latex_metrics_table(gm: Dict[str, np.ndarray]) -> str:
    rows = []
    for t in range(N_STEPS + 1):
        cov = gm["coverage"][:, t].mean()
        rows.append(
            f"        {t:>2} & {_ci_cell(gm['readiness'], t)} & {cov:.2f}"
            f" & {_ci_cell(gm['cost'], t)} & {_ci_cell(gm['efficiency'], t)} \\\\"
        )
    body = "\n".join(rows)
    return (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        rf"\caption{{Greedy placement baseline metrics over {N_STEPS} time steps "
        rf"({int(CONFIDENCE*100)}\% CI across {len(SEEDS)} random seeds; "
        rf"$\delta = {DEGRADATION_RATE}$; location capacity $= {LOCATION_CAPACITY}$). "
        r"Values shown as mean~(lower, upper).}" "\n"
        r"\label{tab:exp1_metrics}" "\n"
        r"\begin{tabular}{ccccc}" "\n"
        r"\hline" "\n"
        r"Step & Readiness & Coverage & Sustainment Cost & Posture Efficiency \\" "\n"
        r"\hline" "\n"
        f"{body}\n"
        r"\hline" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


def latex_sensitivity_table(threat: Dict[str, np.ndarray]) -> str:
    steps_shown = [0, 2, 4, 6, 8, 10]
    rows = []
    for t in steps_shown:
        mu_u  = threat["uniform"][:, t].mean()
        mu_s  = threat["skewed"][:, t].mean()
        delta = mu_u - mu_s
        pct   = (delta / mu_u * 100) if mu_u > 0 else 0.0
        rows.append(
            f"        {t:>2} & {_ci_cell(threat['uniform'], t)}"
            f" & {_ci_cell(threat['skewed'], t)}"
            f" & {delta:.3f} & {pct:.1f}\\% \\\\"
        )
    body = "\n".join(rows)
    return (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        rf"\caption{{Greedy scenario-weighted readiness (SWR) under uniform and skewed "
        rf"threat distributions ({int(CONFIDENCE*100)}\% CI across {len(SEEDS)} random seeds; "
        rf"{N_THREAT_SCENARIOS} scenarios per condition). "
        r"$\Delta = \text{SWR}_{\text{uniform}} - \text{SWR}_{\text{skewed}}$; "
        r"\%~drop $= \Delta / \text{SWR}_{\text{uniform}}$. "
        r"Values shown as mean~(lower, upper).}" "\n"
        r"\label{tab:exp1_sensitivity}" "\n"
        r"\begin{tabular}{ccccc}" "\n"
        r"\hline" "\n"
        r"Step & SWR (uniform) & SWR (skewed) & $\Delta$ & \% Drop \\" "\n"
        r"\hline" "\n"
        f"{body}\n"
        r"\hline" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


def _fmt_ttest_row(label: str, r: dict) -> str:
    """Format one significance-table row, handling NaN/inf gracefully."""
    diff_str = f"{r['mean_diff']:+.4f}"
    d = r.get("cohens_d", float("nan"))
    pwr = r.get("power", float("nan"))

    if math.isnan(r["t_stat"]):
        # Placement-invariant: differences identically zero
        d_str = r"n/a$^{\dagger}$"
        return (
            f"        {label} & {diff_str} & {d_str} & "
            r"\multicolumn{2}{c}{n/a$^{\dagger}$} & n/a \\"
        )

    # Cohen's d display
    if math.isinf(d):
        d_str = r"$-\infty$" if d < 0 else r"$+\infty$"
    else:
        d_str = f"{d:+.2f}"

    star = "$^{*}$" if r["reject_null"] else ""
    p_str = "$<$0.0001" if r["p_value"] < 0.0001 else f"{r['p_value']:.4f}"
    t_val = f"{r['t_stat']:+.3f}" if not math.isinf(r["t_stat"]) else r"$-\infty$"
    pwr_str = f"{pwr:.3f}" if not math.isnan(pwr) else "---"
    return (
        f"        {label} & {diff_str} & {d_str} & {t_val} & {p_str}{star} & {pwr_str} \\\\"
    )


def latex_significance_table(
    sig: Dict[str, dict], decomp: Dict[str, dict]
) -> str:
    adj_alpha = bonferroni_correct(1 - CONFIDENCE, N_COMPARISONS)
    metric_labels = {
        "readiness":   r"Readiness (greedy vs.\ random)",
        "coverage":    r"Coverage (greedy vs.\ random)",
        "cost":        r"Sustainment cost (greedy vs.\ random)",
        "efficiency":  r"Posture efficiency (greedy vs.\ random)",
        "swr_uniform": r"SWR: uniform vs.\ skewed threat",
    }
    sig_rows = "\n".join(
        _fmt_ttest_row(label, sig[key])
        for key, label in metric_labels.items()
    )

    dec_rows = "\n".join(
        f"        {lbl} & {decomp[c]['outer_var']:.5f} & {decomp[c]['inner_var']:.5f}"
        f" & {decomp[c]['total_var']:.5f} & {decomp[c]['icc']:.3f} & \\\\"
        for c, lbl in (("uniform", "Uniform threat"), ("skewed", "Skewed threat"))
    )

    n_outer = decomp["uniform"]["n_outer"]
    n_inner = decomp["uniform"]["n_inner"]

    return (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        rf"\caption{{Statistical significance, effect sizes, and post-hoc power at $t = {N_STEPS}$. "
        r"\textbf{Top}: paired $t$-tests (two-tailed) comparing greedy vs.\ random "
        r"on each metric, and uniform vs.\ skewed SWR. "
        rf"Bonferroni-corrected $\alpha^* = {adj_alpha:.4f}$ ({N_COMPARISONS} comparisons). "
        r"Cohen's $d$ = mean diff.\ / SD of differences; $|d| \geq 0.8$ is a large effect. "
        r"Power computed via the noncentral $t$-distribution ($\lambda = |d|\sqrt{n}$). "
        r"$^{{*}}$ denotes $p < \alpha^*$. "
        r"$^{{\dagger}}$ metric is placement-invariant (zero variance in paired differences). "
        r"\textbf{Bottom}: two-level variance decomposition of SWR "
        rf"({n_outer} scenario seeds $\times$ {n_inner} simulation seeds). "
        r"ICC near 0 = initial-state randomness dominates; "
        r"ICC near 1 = scenario-set choice dominates.}" "\n"
        r"\label{tab:exp1_significance}" "\n"
        r"\begin{tabular}{lccccc}" "\n"
        r"\hline" "\n"
        r"Metric & Mean diff. & Cohen's $d$ & $t$-stat & $p$-value & Power \\" "\n"
        r"\hline" "\n"
        f"{sig_rows}\n"
        r"\hline" "\n"
        r"\multicolumn{6}{l}{\textit{Variance decomposition of SWR}} \\" "\n"
        r"\hline" "\n"
        r"Condition & Outer var. & Inner var. & Total var. & ICC & \\" "\n"
        r"\hline" "\n"
        f"{dec_rows}\n"
        r"\hline" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_summary(gm, threat, sig, decomp):
    W = 76
    sep = "=" * W

    print(f"\n{sep}")
    print("  EXPERIMENT 1 — GREEDY BASELINE CHARACTERIZATION")
    print(f"  {N_ASSETS} assets | {N_LOCATIONS} locs (cap {LOCATION_CAPACITY}) | "
          f"{N_STEPS} steps | δ={DEGRADATION_RATE} | {len(SEEDS)} sim seeds × "
          f"{len(SCENARIO_SEEDS)} scenario seeds")
    print(sep)

    print(f"\n  Per-step metrics  [{int(CONFIDENCE*100)}% CI = mean (lower, upper)]")
    print(f"  {'t':>2}  {'Readiness':>22}  {'Cov':>4}  "
          f"{'Cost':>22}  {'Efficiency':>22}")
    print("  " + "-" * 78)
    for t in range(N_STEPS + 1):
        ci_r = confidence_interval(gm["readiness"][:, t], CONFIDENCE)
        ci_k = confidence_interval(gm["cost"][:, t],      CONFIDENCE)
        ci_e = confidence_interval(gm["efficiency"][:, t], CONFIDENCE)
        cov  = gm["coverage"][:, t].mean()
        print(f"  {t:>2}  "
              f"{ci_r['mean']:.3f} ({ci_r['lower']:.3f},{ci_r['upper']:.3f})  "
              f"{cov:.2f}  "
              f"{ci_k['mean']:.2f} ({ci_k['lower']:.2f},{ci_k['upper']:.2f})  "
              f"{ci_e['mean']:.4f} ({ci_e['lower']:.4f},{ci_e['upper']:.4f})")

    print(f"\n{sep}")
    print("  SIGNIFICANCE TESTS  (greedy vs. random, t=10, Bonferroni-corrected)")
    adj = bonferroni_correct(1 - CONFIDENCE, N_COMPARISONS)
    print(f"  Adjusted α = {adj:.4f}  ({N_COMPARISONS} comparisons)")
    print(sep)
    labels = {"readiness": "Readiness", "coverage": "Coverage",
              "cost": "Cost", "efficiency": "Efficiency",
              "swr_uniform": "SWR uniform vs. skewed"}
    for key, label in labels.items():
        r = sig[key]
        flag = " *" if r["reject_null"] else ""
        print(f"  {label:<28}  diff={r['mean_diff']:+.4f}  "
              f"t={r['t_stat']:+.3f}  p={r['p_value']:.4f}{flag}")

    print(f"\n{sep}")
    print(f"  VARIANCE DECOMPOSITION  (SWR at t={N_STEPS}, "
          f"{len(SCENARIO_SEEDS)} scenario seeds × {len(SEEDS)} sim seeds)")
    print(sep)
    print(f"  {'Condition':<16}  {'Outer var':>10}  {'Inner var':>10}  "
          f"{'Total var':>10}  {'ICC':>6}")
    print("  " + "-" * 56)
    for cond in ("uniform", "skewed"):
        d = decomp[cond]
        print(f"  {cond:<16}  {d['outer_var']:>10.5f}  {d['inner_var']:>10.5f}  "
              f"{d['total_var']:>10.5f}  {d['icc']:>6.3f}")
    print(f"\n  ICC near 0 → initial-state randomness dominates.")
    print(f"  ICC near 1 → scenario-set choice dominates.")

    print(f"\n{sep}")
    print(f"  THREAT SENSITIVITY  [{int(CONFIDENCE*100)}% CI]")
    print(sep)
    print(f"  {'t':>2}  {'SWR uniform':>24}  {'SWR skewed':>24}  {'Δ':>7}  {'%drop':>6}")
    print("  " + "-" * 72)
    for t in range(N_STEPS + 1):
        ci_u = confidence_interval(threat["uniform"][:, t], CONFIDENCE)
        ci_s = confidence_interval(threat["skewed"][:, t],  CONFIDENCE)
        delta = ci_u["mean"] - ci_s["mean"]
        pct   = (delta / ci_u["mean"] * 100) if ci_u["mean"] > 0 else 0.0
        print(f"  {t:>2}  "
              f"{ci_u['mean']:.3f} ({ci_u['lower']:.3f},{ci_u['upper']:.3f})  "
              f"{ci_s['mean']:.3f} ({ci_s['lower']:.3f},{ci_s['upper']:.3f})  "
              f"{delta:>+7.3f}  {pct:>5.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Running Experiment 1  "
          f"({len(SEEDS)} sim seeds × {len(SCENARIO_SEEDS)} scenario seeds) ...")

    gm     = collect_metrics(SEEDS, use_greedy=True)
    rm     = collect_metrics(SEEDS, use_greedy=False)
    threat = collect_threat_sensitivity(SEEDS)

    print("  Computing significance tests ...")
    sig = compute_significance(gm, rm, threat)

    print("  Running variance decomposition ...")
    decomp = collect_variance_decomposition(SEEDS, SCENARIO_SEEDS)

    print_summary(gm, threat, sig, decomp)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    fig = make_figure(gm, rm, threat)
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"exp1_main.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"\n  Figure  → {p}")
    plt.close(fig)

    tables = {
        "exp1_metrics.tex":      latex_metrics_table(gm),
        "exp1_sensitivity.tex":  latex_sensitivity_table(threat),
        "exp1_significance.tex": latex_significance_table(sig, decomp),
    }
    for fname, content in tables.items():
        p = TAB_DIR / fname
        p.write_text(content)
        print(f"  Table   → {p}")

    print()
    for content in tables.values():
        print(content)
        print()


if __name__ == "__main__":
    main()
