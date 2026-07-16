"""Tests for Experiment 3: Adversarial Bayesian counter-move model and RobustCEVOptimizer."""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from postureopt.core import Asset, AssetType, Location, PostureOptimizer, ThreatScenario
from postureopt.drsp import AdversarialModel, RobustCEVOptimizer
from postureopt.evaluate import scenario_weighted_readiness


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _locs():
    return [
        Location("L1", "Alpha",   26.0, 127.0, capacity=10, strategic_value=1.00),
        Location("L2", "Beta",    34.0, 132.0, capacity=10, strategic_value=0.70),
        Location("L3", "Gamma",   13.0, 144.0, capacity=10, strategic_value=0.60),
        Location("L4", "Delta",   40.0, 141.0, capacity=10, strategic_value=0.50),
        Location("L5", "Epsilon", 37.0, 127.0, capacity=10, strategic_value=0.40),
    ]


def _assets(n: int = 10) -> list:
    return [Asset(f"A{i:03d}", AssetType.AIRCRAFT, "L1", quantity=2, readiness_rate=0.85)
            for i in range(n)]


def _apply(assets, assignment):
    return [
        Asset(a.asset_id, a.asset_type,
              assignment.get(a.asset_id, a.location_id),
              a.quantity, a.readiness_rate, a.maintenance_days_remaining)
        for a in assets
    ]


def _prior():
    """90% safe, 10% L1 attacked — naive CEV picks L1 (highest effective value)."""
    return [
        ThreatScenario("S_safe",   {},           weight=9.0),
        ThreatScenario("S_atk_L1", {"L1": 0.99}, weight=1.0),
    ]


# ---------------------------------------------------------------------------
# AdversarialModel — construction
# ---------------------------------------------------------------------------

def test_adversarial_model_rejects_bad_p_obs():
    with pytest.raises(ValueError):
        AdversarialModel(p_obs=1.5)


def test_adversarial_model_rejects_bad_rationality():
    with pytest.raises(ValueError):
        AdversarialModel(p_obs=0.5, rationality=-0.1)


# ---------------------------------------------------------------------------
# AdversarialModel — weight updates
# ---------------------------------------------------------------------------

def test_zero_p_obs_leaves_weights_unchanged():
    locs = _locs()
    assets = _assets(10)
    scenarios = _prior()
    adversary = AdversarialModel(p_obs=0.0)
    assignment = {a.asset_id: "L1" for a in assets}
    updated = adversary.update_weights(scenarios, assignment, locs)
    for orig, upd in zip(scenarios, updated):
        assert orig.weight == pytest.approx(upd.weight)


def test_full_obs_shifts_weight_toward_threatened_location():
    """With p_obs=1, all assets at L1: adversary concentrates on S_atk_L1."""
    locs = _locs()
    assets = _assets(10)
    scenarios = _prior()
    adversary = AdversarialModel(p_obs=1.0, rationality=1.0)
    assignment = {a.asset_id: "L1" for a in assets}
    updated = adversary.update_weights(scenarios, assignment, locs)
    # S_atk_L1 targets L1 heavily; its weight must increase
    assert updated[1].weight > scenarios[1].weight


def test_total_weight_preserved_after_update():
    locs = _locs()
    assets = _assets(10)
    scenarios = _prior()
    adversary = AdversarialModel(p_obs=0.75, rationality=0.8)
    assignment = {a.asset_id: "L2" for a in assets}
    updated = adversary.update_weights(scenarios, assignment, locs)
    assert sum(s.weight for s in updated) == pytest.approx(
        sum(s.weight for s in scenarios), rel=1e-9
    )


def test_no_adversarial_signal_returns_unchanged():
    """If no scenario has any threat at the occupied location, weights unchanged."""
    locs = _locs()
    scenarios = [
        ThreatScenario("S1", {"L3": 0.9}, weight=1.0),  # threats L3, not L2
        ThreatScenario("S2", {"L4": 0.9}, weight=1.0),  # threats L4, not L2
    ]
    adversary = AdversarialModel(p_obs=1.0, rationality=1.0)
    assignment = {"A000": "L2"}  # assets at L2, unthreatened by any scenario
    updated = adversary.update_weights(scenarios, assignment, locs)
    for orig, upd in zip(scenarios, updated):
        assert orig.weight == pytest.approx(upd.weight)


# ---------------------------------------------------------------------------
# RobustCEVOptimizer — structure
# ---------------------------------------------------------------------------

def test_robust_cev_returns_full_assignment():
    locs = _locs()
    assets = _assets(10)
    robust = RobustCEVOptimizer(AdversarialModel(p_obs=0.5))
    assignment, final = robust.optimize_placement(assets, locs, _prior())
    assert len(assignment) == len(assets)
    assert all(v in {loc.location_id for loc in locs} for v in assignment.values())


def test_robust_cev_total_weight_preserved():
    locs = _locs()
    assets = _assets(10)
    scenarios = _prior()
    robust = RobustCEVOptimizer(AdversarialModel(p_obs=0.6))
    _, final = robust.optimize_placement(assets, locs, scenarios)
    assert sum(s.weight for s in final) == pytest.approx(
        sum(s.weight for s in scenarios), rel=1e-9
    )


# ---------------------------------------------------------------------------
# Experiment 3 core claim: robust floor > naive under full adversarial observation
# ---------------------------------------------------------------------------

def test_robust_maintains_floor_vs_naive_under_full_adversarial_observation():
    """Key Experiment 3 result.

    Prior: 90% safe, 10% L1 attacked. Adversary observes perfectly (p_obs=1).
    Naive CEV: places at L1 (highest expected value under prior) → adversary
    concentrates attack → SWR collapses.
    Robust CEV: iterates away from L1 → adversary has no concentrated target →
    SWR holds up. Robust SWR must exceed naive SWR under adversarial conditions.
    """
    locs = _locs()
    assets = _assets(10)
    prior = _prior()
    adversary = AdversarialModel(p_obs=1.0, rationality=1.0)

    # Naive CEV: optimises against prior, unaware of adversarial response
    from postureopt.core import ScenarioWeightedOptimizer
    naive_cev = ScenarioWeightedOptimizer(seed=42)
    naive_assignment = naive_cev.optimize_placement(assets, locs, prior)
    adv_for_naive = adversary.update_weights(prior, naive_assignment, locs)
    naive_swr = scenario_weighted_readiness(_apply(assets, naive_assignment), adv_for_naive)

    # Robust CEV: accounts for adversarial observation via iteration
    robust = RobustCEVOptimizer(adversary, seed=42)
    robust_assignment, adv_for_robust = robust.optimize_placement(assets, locs, prior)
    robust_swr = scenario_weighted_readiness(_apply(assets, robust_assignment), adv_for_robust)

    assert robust_swr > naive_swr, (
        f"Robust SWR ({robust_swr:.4f}) must exceed naive SWR ({naive_swr:.4f}) "
        "under full adversarial observation"
    )


def test_zero_obs_no_gap_between_robust_and_naive():
    """With p_obs=0 the adversary cannot observe: robust and naive converge to same placement."""
    locs = _locs()
    assets = _assets(10)
    prior = _prior()
    adversary = AdversarialModel(p_obs=0.0)

    from postureopt.core import ScenarioWeightedOptimizer
    naive_assignment = ScenarioWeightedOptimizer(seed=42).optimize_placement(assets, locs, prior)
    robust = RobustCEVOptimizer(adversary, seed=42)
    robust_assignment, _ = robust.optimize_placement(assets, locs, prior)

    assert naive_assignment == robust_assignment
