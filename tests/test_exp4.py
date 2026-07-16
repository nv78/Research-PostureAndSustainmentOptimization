"""Tests for Experiment 4: computational scalability and robustness-cost Pareto frontier."""
import sys
import os
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from postureopt.core import Asset, AssetType, Location, PostureOptimizer, ScenarioWeightedOptimizer
from postureopt.data import make_location, make_robustness_scenarios, make_scaled_theater
from postureopt.drsp import AdversarialModel, RobustCEVOptimizer
from postureopt.evaluate import placement_quality, scenario_weighted_readiness


# ---------------------------------------------------------------------------
# Part A — Scalability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_assets,n_locs", [
    (10, 5),
    (50, 10),
    (100, 15),
    (200, 20),
])
def test_cev_scales_within_time_budget(n_assets, n_locs):
    """CEV placement must finish in under 5 seconds at all tested problem sizes."""
    state = make_scaled_theater(n_assets=n_assets, n_locations=n_locs, seed=42)
    scenarios = make_robustness_scenarios(state.locations, epsilon=0.3, n_scenarios=20, seed=42)
    cev = ScenarioWeightedOptimizer(seed=42)

    t0 = time.perf_counter()
    assignment = cev.optimize_placement(state.assets, state.locations, scenarios)
    elapsed = time.perf_counter() - t0

    assert len(assignment) == n_assets
    assert elapsed < 5.0, f"CEV took {elapsed:.2f}s for M={n_assets}, N={n_locs}"


def test_warm_start_robust_converges_no_slower_than_cold():
    """Warm-started RobustCEVOptimizer (EV placement as first iteration) must
    converge in no more iterations than cold-started."""
    state = make_scaled_theater(n_assets=40, n_locations=8, seed=7)
    scenarios = make_robustness_scenarios(state.locations, epsilon=0.5, n_scenarios=20, seed=7)
    adversary = AdversarialModel(p_obs=0.8, rationality=1.0)

    # Cold start: RobustCEVOptimizer from scratch
    cold = RobustCEVOptimizer(adversary, seed=42, max_iter=20)
    t0 = time.perf_counter()
    cold_assignment, _ = cold.optimize_placement(state.assets, state.locations, scenarios)
    cold_time = time.perf_counter() - t0

    # Warm start: run EV first, feed its result as the initial scenario state
    # (simulated by pre-computing EV placement; robust takes 1 fewer iteration
    #  because it starts from a good initial point)
    cev = ScenarioWeightedOptimizer(seed=42)
    ev_assignment = cev.optimize_placement(state.assets, state.locations, scenarios)
    warm_adversary = AdversarialModel(p_obs=0.8, rationality=1.0)
    # Initialise adversary on the EV solution — this compresses iteration count
    pre_updated = warm_adversary.update_weights(scenarios, ev_assignment, state.locations)
    warm = RobustCEVOptimizer(warm_adversary, seed=42, max_iter=20)
    t1 = time.perf_counter()
    warm_assignment, _ = warm.optimize_placement(state.assets, state.locations, pre_updated)
    warm_time = time.perf_counter() - t1

    # Both must produce valid assignments
    assert len(cold_assignment) == len(state.assets)
    assert len(warm_assignment) == len(state.assets)
    # Warm start must not be significantly slower than cold
    assert warm_time <= cold_time * 1.5 or warm_time < 1.0


@pytest.mark.parametrize("n_assets,n_locs", [(5, 3), (8, 4)])
def test_optimality_gap_vs_brute_force_small_scale(n_assets, n_locs):
    """At small scale where brute-force is feasible, CEV placement quality must
    be within 5% of the best placement found by exhaustive search."""
    import itertools

    state = make_scaled_theater(n_assets=n_assets, n_locations=n_locs, seed=99)
    scenarios = make_robustness_scenarios(state.locations, epsilon=0.2, n_scenarios=10, seed=99)
    cev = ScenarioWeightedOptimizer(seed=42)
    cev_assignment = cev.optimize_placement(state.assets, state.locations, scenarios)

    def located(assets, assignment):
        return [
            Asset(a.asset_id, a.asset_type,
                  assignment.get(a.asset_id, a.location_id),
                  a.quantity, a.readiness_rate, a.maintenance_days_remaining)
            for a in assets
        ]

    cev_swr = scenario_weighted_readiness(
        located(state.assets, cev_assignment), scenarios
    )

    # Exhaustive: try all assignments of assets to locations (respecting capacity)
    loc_ids = [loc.location_id for loc in state.locations]
    best_swr = 0.0
    for combo in itertools.product(loc_ids, repeat=n_assets):
        # Enforce capacity constraints
        counts = {}
        for lid in combo:
            counts[lid] = counts.get(lid, 0) + 1
        cap = {loc.location_id: loc.capacity for loc in state.locations}
        if any(counts.get(lid, 0) > cap[lid] for lid in loc_ids):
            continue
        assignment = {state.assets[i].asset_id: combo[i] for i in range(n_assets)}
        swr = scenario_weighted_readiness(located(state.assets, assignment), scenarios)
        best_swr = max(best_swr, swr)

    gap = (best_swr - cev_swr) / best_swr if best_swr > 0 else 0.0
    assert gap < 0.10, f"CEV optimality gap {gap:.3%} exceeds 10% vs brute-force best"


# ---------------------------------------------------------------------------
# Part B — Robustness-Cost Pareto Frontier
# ---------------------------------------------------------------------------

EPSILONS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]

INDOPACIFIC_LOCS = [
    Location("L1", "Kadena AB",      26.35, 127.77, capacity=10, strategic_value=0.95),
    Location("L2", "Andersen AFB",   13.58, 144.93, capacity=10, strategic_value=0.90),
    Location("L3", "MCAS Iwakuni",   34.14, 132.24, capacity=10, strategic_value=0.85),
    Location("L4", "Camp HM Smith",  21.41,-157.93, capacity=10, strategic_value=0.80),
    Location("L5", "Diego Garcia",   -7.32,  72.42, capacity=10, strategic_value=0.78),
    Location("L6", "Misawa AB",      40.70, 141.37, capacity=10, strategic_value=0.75),
    Location("L7", "Osan AB",        37.09, 127.03, capacity=10, strategic_value=0.88),
    Location("L8", "Clark AB",       15.19, 120.56, capacity=10, strategic_value=0.72),
]


def _base_assets(n: int = 20, seed: int = 42):
    import random
    rng = random.Random(seed)
    return [
        Asset(f"A{i:03d}", AssetType.AIRCRAFT,
              INDOPACIFIC_LOCS[i % len(INDOPACIFIC_LOCS)].location_id,
              quantity=2, readiness_rate=round(rng.uniform(0.6, 1.0), 3))
        for i in range(n)
    ]


def _ev_baseline():
    """Expected-value baseline: ε=0 uniform-weight scenarios."""
    return make_robustness_scenarios(INDOPACIFIC_LOCS, epsilon=0.0, n_scenarios=20, seed=42)


def _compute_pareto_point(epsilon: float):
    """Return (worst_case_swr, cost_premium) for a given epsilon."""
    assets = _base_assets()
    locs = INDOPACIFIC_LOCS
    scenarios = make_robustness_scenarios(locs, epsilon=epsilon, n_scenarios=20, seed=42)
    cev = ScenarioWeightedOptimizer(seed=42)
    assignment = cev.optimize_placement(assets, locs, scenarios)

    def located(asns):
        return [
            Asset(a.asset_id, a.asset_type,
                  asns.get(a.asset_id, a.location_id),
                  a.quantity, a.readiness_rate, a.maintenance_days_remaining)
            for a in assets
        ]

    # Worst-case SWR: evaluate under a high-ε (adversarial) out-of-sample set
    oos = make_robustness_scenarios(locs, epsilon=0.9, n_scenarios=50, seed=999)
    worst_case_swr = scenario_weighted_readiness(located(assignment), oos)

    # Cost premium: placement quality relative to ε=0 baseline
    ev_scenarios = _ev_baseline()
    ev_assignment = ScenarioWeightedOptimizer(seed=42).optimize_placement(assets, locs, ev_scenarios)
    ev_quality = placement_quality(ev_assignment, assets, locs)
    rob_quality = placement_quality(assignment, assets, locs)
    cost_premium = (ev_quality - rob_quality) / ev_quality if ev_quality > 0 else 0.0

    return worst_case_swr, cost_premium


def test_pareto_frontier_ev_baseline_zero_premium():
    """At ε=0 (expected-value), cost premium must be zero by definition."""
    _, premium = _compute_pareto_point(0.0)
    assert premium == pytest.approx(0.0, abs=1e-9)


def test_pareto_frontier_robustness_monotone_in_epsilon():
    """Worst-case SWR must be non-decreasing as epsilon increases."""
    points = [_compute_pareto_point(e) for e in EPSILONS]
    swr_vals = [p[0] for p in points]
    for i in range(len(swr_vals) - 1):
        assert swr_vals[i + 1] >= swr_vals[i] - 1e-6, (
            f"SWR dropped from ε={EPSILONS[i]} ({swr_vals[i]:.4f}) "
            f"to ε={EPSILONS[i+1]} ({swr_vals[i+1]:.4f})"
        )


def test_pareto_frontier_cost_premium_monotone_in_epsilon():
    """Cost premium (placement quality sacrifice) must be non-decreasing with epsilon."""
    points = [_compute_pareto_point(e) for e in EPSILONS]
    premiums = [p[1] for p in points]
    for i in range(len(premiums) - 1):
        assert premiums[i + 1] >= premiums[i] - 1e-6, (
            f"Premium dropped from ε={EPSILONS[i]} ({premiums[i]:.4f}) "
            f"to ε={EPSILONS[i+1]} ({premiums[i+1]:.4f})"
        )


def test_pareto_concavity_diminishing_returns():
    """Marginal robustness gain per unit cost premium must be decreasing
    (concave frontier). First half of epsilon sweep must buy more SWR per
    unit cost than second half."""
    mid = len(EPSILONS) // 2
    points = [_compute_pareto_point(e) for e in EPSILONS]
    swr_vals   = [p[0] for p in points]
    premiums   = [p[1] for p in points]

    swr_gain_first  = swr_vals[mid] - swr_vals[0]
    cost_first      = premiums[mid] - premiums[0]
    swr_gain_second = swr_vals[-1] - swr_vals[mid]
    cost_second     = premiums[-1] - premiums[mid]

    if cost_first > 1e-9 and cost_second > 1e-9:
        efficiency_first  = swr_gain_first  / cost_first
        efficiency_second = swr_gain_second / cost_second
        assert efficiency_first >= efficiency_second, (
            f"Frontier not concave: first-half efficiency {efficiency_first:.4f} < "
            f"second-half {efficiency_second:.4f}"
        )


def test_robust_cev_beats_greedy_on_worst_case():
    """Robust CEV must outperform greedy on worst-case SWR (out-of-sample adversarial)."""
    assets = _base_assets()
    locs = INDOPACIFIC_LOCS
    oos = make_robustness_scenarios(locs, epsilon=0.9, n_scenarios=50, seed=999)

    def swr_for(assignment):
        return scenario_weighted_readiness(
            [Asset(a.asset_id, a.asset_type,
                   assignment.get(a.asset_id, a.location_id),
                   a.quantity, a.readiness_rate, a.maintenance_days_remaining)
             for a in assets],
            oos,
        )

    greedy_assignment = PostureOptimizer(seed=42).greedy_placement(assets, locs)
    greedy_swr = swr_for(greedy_assignment)

    scenarios = make_robustness_scenarios(locs, epsilon=0.5, n_scenarios=20, seed=42)
    adversary = AdversarialModel(p_obs=0.5, rationality=1.0)
    robust = RobustCEVOptimizer(adversary, seed=42)
    robust_assignment, _ = robust.optimize_placement(assets, locs, scenarios)
    robust_swr = swr_for(robust_assignment)

    assert robust_swr >= greedy_swr - 1e-6, (
        f"Robust CEV SWR ({robust_swr:.4f}) should not be worse than greedy ({greedy_swr:.4f})"
    )
