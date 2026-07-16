"""Tests for Experiment 5: dual-theater case studies, minimax, A2/AD, and sensitivity."""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from postureopt.core import Asset, AssetType, Location, PostureOptimizer, ScenarioWeightedOptimizer, ThreatScenario
from postureopt.data import (
    make_indopacific_theater,
    make_european_theater,
    make_indopacific_scenarios,
    make_european_scenarios,
    make_a2ad_scenarios,
    A2AD_THREAT_CENTER,
    _INDOPACIFIC_LOCATIONS,
    _EUROPEAN_LOCATIONS,
)
from postureopt.drsp import AdversarialModel, MinimaxOptimizer, RobustCEVOptimizer
from postureopt.evaluate import (
    assignment_stability,
    contested_zone_coverage,
    scenario_weighted_readiness,
)


# ---------------------------------------------------------------------------
# Theater construction
# ---------------------------------------------------------------------------

def test_indopacific_theater_asset_counts():
    state = make_indopacific_theater()
    assert len(state.assets) == 20
    assert len(state.locations) == 8
    by_type = {}
    for a in state.assets:
        by_type[a.asset_type] = by_type.get(a.asset_type, 0) + 1
    assert by_type[AssetType.AIRCRAFT] == 12
    assert by_type[AssetType.MAINTENANCE_CREW] == 2
    assert by_type[AssetType.MUNITION] == 4
    assert by_type[AssetType.FUEL_DEPOT] == 2


def test_european_theater_asset_counts():
    state = make_european_theater()
    assert len(state.assets) == 15
    assert len(state.locations) == 6
    by_type = {}
    for a in state.assets:
        by_type[a.asset_type] = by_type.get(a.asset_type, 0) + 1
    assert by_type[AssetType.AIRCRAFT] == 8
    assert by_type[AssetType.MUNITION] == 4
    assert by_type[AssetType.FUEL_DEPOT] == 3


# ---------------------------------------------------------------------------
# Theater scenarios
# ---------------------------------------------------------------------------

def test_indopacific_scenarios_count():
    scenarios = make_indopacific_scenarios(n_scenarios=20, seed=42)
    assert len(scenarios) == 20
    assert all(s.weight == 1.0 for s in scenarios)
    assert all("L1" in s.threat_levels for s in scenarios)


def test_indopacific_scenarios_threat_range():
    scenarios = make_indopacific_scenarios(n_scenarios=50, seed=0)
    for s in scenarios:
        for threat in s.threat_levels.values():
            assert 0.0 <= threat <= 1.0


def test_european_scenarios_all_locations_covered():
    scenarios = make_european_scenarios(n_scenarios=20, seed=42)
    loc_ids = {loc.location_id for loc in _EUROPEAN_LOCATIONS}
    for s in scenarios:
        assert set(s.threat_levels.keys()) == loc_ids


# ---------------------------------------------------------------------------
# MinimaxOptimizer
# ---------------------------------------------------------------------------

def test_minimax_returns_full_assignment():
    state = make_indopacific_theater()
    scenarios = make_indopacific_scenarios(n_scenarios=10, seed=42)
    mm = MinimaxOptimizer(seed=42)
    assignment = mm.optimize_placement(state.assets, state.locations, scenarios)
    assert len(assignment) == len(state.assets)
    loc_ids = {loc.location_id for loc in state.locations}
    assert all(v in loc_ids for v in assignment.values())


def test_minimax_avoids_highest_threat_location():
    """Given one location with worst-case threat=0.99, minimax must prefer others."""
    locs = [
        Location("LA", "High-threat", 0.0, 0.0, capacity=10, strategic_value=0.95),
        Location("LB", "Safe",        1.0, 0.0, capacity=10, strategic_value=0.80),
    ]
    assets = [Asset(f"A{i}", AssetType.AIRCRAFT, "LA", quantity=2, readiness_rate=0.85)
              for i in range(4)]
    scenarios = [ThreatScenario("S0", {"LA": 0.99, "LB": 0.05}, weight=1.0)]
    mm = MinimaxOptimizer(seed=42)
    assignment = mm.optimize_placement(assets, locs, scenarios)
    # All 4 assets fit in LB (capacity=10); minimax should prefer LB
    assert all(v == "LB" for v in assignment.values())


# ---------------------------------------------------------------------------
# A2/AD scenarios
# ---------------------------------------------------------------------------

def test_a2ad_inside_radius_has_high_threat():
    """Location at distance < radius must receive threat > 0.4."""
    locs = [
        Location("CLOSE", "Close", 30.0, 130.5, capacity=10, strategic_value=0.9),
        Location("FAR",   "Far",   10.0,  70.0, capacity=10, strategic_value=0.8),
    ]
    scenarios = make_a2ad_scenarios(locs, 30.0, 130.0, a2ad_radius_km=100.0,
                                    n_scenarios=20, seed=42)
    mean_close = sum(s.threat_levels["CLOSE"] for s in scenarios) / len(scenarios)
    mean_far   = sum(s.threat_levels["FAR"]   for s in scenarios) / len(scenarios)
    assert mean_close > 0.4
    assert mean_far < 0.3


def test_a2ad_inside_exceeds_outside():
    """Mean threat inside radius must exceed mean threat outside."""
    state = make_indopacific_theater()
    lat, lon = A2AD_THREAT_CENTER
    scenarios = make_a2ad_scenarios(state.locations, lat, lon, a2ad_radius_km=500.0,
                                    n_scenarios=30, seed=7)
    from postureopt.core import haversine_distance
    inside, outside = [], []
    for s in scenarios:
        for loc in state.locations:
            d = haversine_distance(lat, lon, loc.lat, loc.lon)
            t = s.threat_levels[loc.location_id]
            (inside if d <= 500.0 else outside).append(t)
    if inside and outside:
        assert sum(inside) / len(inside) > sum(outside) / len(outside)


# ---------------------------------------------------------------------------
# assignment_stability
# ---------------------------------------------------------------------------

def test_assignment_stability_identical():
    a = {"A0": "L1", "A1": "L2", "A2": "L1"}
    assert assignment_stability(a, a) == pytest.approx(1.0)


def test_assignment_stability_disjoint():
    a = {"A0": "L1", "A1": "L1"}
    b = {"A0": "L2", "A1": "L2"}
    assert assignment_stability(a, b) == pytest.approx(0.0)


def test_assignment_stability_partial():
    a = {"A0": "L1", "A1": "L2", "A2": "L1"}
    b = {"A0": "L1", "A1": "L1", "A2": "L2"}
    # A0 matches, A1 and A2 differ -> 1/3
    assert assignment_stability(a, b) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# contested_zone_coverage
# ---------------------------------------------------------------------------

def test_contested_zone_coverage_all_outside():
    """All assets far from threat center -> coverage = 1.0."""
    state = make_indopacific_theater()
    assignment = {a.asset_id: "L4" for a in state.assets}  # Camp HM Smith ~6000km away
    lat, lon = A2AD_THREAT_CENTER
    cov = contested_zone_coverage(assignment, state.locations, lat, lon, radius_km=600.0)
    assert cov == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Core Experiment 5 claim: DRSO more stable than EV under scenario perturbation
# ---------------------------------------------------------------------------

def test_drso_stability_exceeds_ev_under_perturbation():
    """Key Exp 5 result: DRSO assignment is more stable than EV under ±30% weight perturbation."""
    import random
    state = make_indopacific_theater()
    locs = state.locations
    base_scenarios = make_indopacific_scenarios(n_scenarios=20, seed=42)

    # EV baseline
    ev = ScenarioWeightedOptimizer(seed=42)
    ev_base = ev.optimize_placement(state.assets, locs, base_scenarios)

    # DRSO baseline
    adversary = AdversarialModel(p_obs=0.7, rationality=1.0)
    drso = RobustCEVOptimizer(adversary, seed=42, max_iter=10)
    drso_base, _ = drso.optimize_placement(state.assets, locs, base_scenarios)

    # Perturb weights ±30%
    rng = random.Random(99)
    perturbed = [
        ThreatScenario(s.scenario_id, s.threat_levels,
                       weight=s.weight * rng.uniform(0.70, 1.30))
        for s in base_scenarios
    ]

    ev_perturbed = ev.optimize_placement(state.assets, locs, perturbed)
    drso_perturbed, _ = RobustCEVOptimizer(
        AdversarialModel(p_obs=0.7, rationality=1.0), seed=42, max_iter=10
    ).optimize_placement(state.assets, locs, perturbed)

    drso_stab = assignment_stability(drso_base, drso_perturbed)
    ev_stab   = assignment_stability(ev_base,   ev_perturbed)
    assert drso_stab >= ev_stab, (
        f"DRSO stability ({drso_stab:.3f}) should be >= EV stability ({ev_stab:.3f}) "
        "under ±30% scenario weight perturbation"
    )
