"""Evaluation metrics for postureopt."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .core import (
    Asset,
    Location,
    PostureState,
    ResourceProfile,
    SLATarget,
    SustainmentAction,
    TelemetrySnapshot,
    ThreatScenario,
    estimated_hourly_cost,
    estimated_latency_ms,
    resource_utilization_score,
)

ACTION_COSTS: Dict[str, float] = {
    "REPOSITION": 10.0,
    "RESUPPLY": 5.0,
    "MAINTAIN": 2.0,
    "HOLD": 0.0,
}


def readiness_score(assets: List[Asset]) -> float:
    """Quantity-weighted mean readiness rate."""
    total_qty = sum(a.quantity for a in assets)
    if total_qty == 0:
        return 0.0
    return sum(a.readiness_rate * a.quantity for a in assets) / total_qty


def coverage_score(assets: List[Asset], locations: List[Location]) -> float:
    """Fraction of locations with at least one asset."""
    if not locations:
        return 0.0
    covered = {a.location_id for a in assets}
    loc_ids = {loc.location_id for loc in locations}
    return len(covered & loc_ids) / len(loc_ids)


def sustainment_cost(actions: List[SustainmentAction]) -> float:
    """Sum of action costs."""
    return sum(ACTION_COSTS.get(a.value, 0.0) for a in actions)


def posture_efficiency(readiness: float, coverage: float, cost: float) -> float:
    """(readiness * coverage) / log(2 + cost). Offset avoids division by zero when cost=0."""
    return (readiness * coverage) / math.log1p(cost + 1.0)


def placement_quality(
    assignments: Dict[str, str],
    assets: List[Asset],
    locations: List[Location],
) -> float:
    """Mean strategic value of assigned locations."""
    loc_map = {loc.location_id: loc.strategic_value for loc in locations}
    values = [loc_map[lid] for lid in assignments.values() if lid in loc_map]
    if not values:
        return 0.0
    return sum(values) / len(values)


def simulation_summary(states: List[PostureState]) -> Dict:
    """Aggregate readiness across simulation steps."""
    if not states:
        return {"n_steps": 0, "mean_readiness": 0.0, "min_readiness": 0.0, "final_readiness": 0.0}
    readiness_vals = [s.total_readiness() for s in states]
    return {
        "n_steps": len(states),
        "mean_readiness": sum(readiness_vals) / len(readiness_vals),
        "min_readiness": min(readiness_vals),
        "final_readiness": readiness_vals[-1],
    }


# ---------------------------------------------------------------------------
# New infrastructure / AI sustainment metrics
# ---------------------------------------------------------------------------


def sla_compliance_score(
    snapshots: List[TelemetrySnapshot],
    sla: SLATarget,
) -> float:
    """Fraction of telemetry snapshots that satisfy all SLA conditions.

    A snapshot is compliant if:
      - p99_latency_ms <= sla.max_latency_ms
      - availability >= sla.min_availability
      - error_rate <= sla.max_error_rate
      - cost_per_hour_usd <= sla.cost_per_hour_usd
    """
    if not snapshots:
        return 0.0
    compliant = sum(
        1
        for s in snapshots
        if (
            s.p99_latency_ms <= sla.max_latency_ms
            and s.availability >= sla.min_availability
            and s.error_rate <= sla.max_error_rate
            and s.cost_per_hour_usd <= sla.cost_per_hour_usd
        )
    )
    return compliant / len(snapshots)


def cost_efficiency_index(
    snapshots: List[TelemetrySnapshot],
    sla: SLATarget,
) -> float:
    """Ratio of compliant throughput to mean cost.

    CEI = (fraction_compliant * mean_rps) / mean_cost_per_hour

    Higher is better: more requests served within SLA per dollar.
    Returns 0 if mean cost is zero.
    """
    if not snapshots:
        return 0.0
    fraction_compliant = sla_compliance_score(snapshots, sla)
    mean_rps = sum(s.requests_per_second for s in snapshots) / len(snapshots)
    mean_cost = sum(s.cost_per_hour_usd for s in snapshots) / len(snapshots)
    if mean_cost == 0.0:
        return 0.0
    return (fraction_compliant * mean_rps) / mean_cost


def sustainability_score(
    snapshots: List[TelemetrySnapshot],
    target_utilization: float = 0.70,
) -> float:
    """Score measuring how close resource utilization is to an efficient target.

    A deployment that is under- or over-utilized wastes resources.
    Score = 1 - |mean_utilization - target| / max(target, 1 - target)
    Returns values in [0, 1].
    """
    if not snapshots:
        return 0.0
    mean_util = resource_utilization_score(snapshots)
    denominator = max(target_utilization, 1.0 - target_utilization)
    return max(0.0, 1.0 - abs(mean_util - target_utilization) / denominator)


def cost_latency_frontier(
    profiles: List[ResourceProfile],
    requests_per_second: float,
    base_latency_ms: float = 10.0,
    gpu_surcharge: float = 2.5,
) -> List[Tuple[float, float]]:
    """Return (cost_per_hour, p99_latency_ms) pairs for each resource profile.

    Useful for visualising the cost-latency Pareto frontier.
    """
    results: List[Tuple[float, float]] = []
    for profile in profiles:
        cost = estimated_hourly_cost(profile, gpu_surcharge)
        latency = estimated_latency_ms(profile, requests_per_second, base_latency_ms)
        results.append((cost, latency))
    return results


def pareto_efficient_profiles(
    profiles: List[ResourceProfile],
    requests_per_second: float,
    base_latency_ms: float = 10.0,
) -> List[ResourceProfile]:
    """Return profiles that are not dominated on (cost, latency).

    A profile is dominated if another has strictly lower cost AND lower latency.
    """
    frontier = cost_latency_frontier(profiles, requests_per_second, base_latency_ms)
    dominated: set = set()
    n = len(profiles)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ci, li = frontier[i]
            cj, lj = frontier[j]
            if cj <= ci and lj <= li and (cj < ci or lj < li):
                dominated.add(i)
    return [p for idx, p in enumerate(profiles) if idx not in dominated]


# ---------------------------------------------------------------------------
# Stochastic optimization metrics (Experiments 1 & 2)
# ---------------------------------------------------------------------------


def scenario_weighted_readiness(
    assets: List[Asset],
    scenarios: List[ThreatScenario],
) -> float:
    """Expected readiness across threat scenarios.

    For each scenario, each asset's effective readiness is readiness_rate
    multiplied by (1 - threat_level at its location).  Returns the
    probability-weighted mean across scenarios.
    """
    if not scenarios or not assets:
        return 0.0
    total_weight = sum(s.weight for s in scenarios)
    if total_weight == 0:
        return 0.0
    total_qty = sum(a.quantity for a in assets)
    if total_qty == 0:
        return 0.0
    expected_r = 0.0
    for s in scenarios:
        scenario_r = sum(
            a.readiness_rate * (1.0 - s.threat_levels.get(a.location_id, 0.0)) * a.quantity
            for a in assets
        ) / total_qty
        expected_r += (s.weight / total_weight) * scenario_r
    return expected_r


def evss(cev_readiness: float, greedy_readiness: float) -> float:
    """Expected Value of the Stochastic Solution (EVSS = CEV - greedy).

    Positive values indicate the scenario-weighted optimizer outperforms greedy.
    """
    return cev_readiness - greedy_readiness


# ---------------------------------------------------------------------------
# Experiment 5 metrics
# ---------------------------------------------------------------------------


def assignment_stability(
    assign_a: Dict[str, str],
    assign_b: Dict[str, str],
) -> float:
    """Fraction of assets with identical location in both assignments (in [0, 1]).

    Returns 1.0 if assignments are identical, 0.0 if every asset moved.
    Only shared asset IDs are compared; unilateral assets are ignored.
    """
    shared = set(assign_a) & set(assign_b)
    if not shared:
        return 1.0 if (not assign_a and not assign_b) else 0.0
    return sum(1 for aid in shared if assign_a[aid] == assign_b[aid]) / len(shared)


def contested_zone_coverage(
    assignment: Dict[str, str],
    locations: List[Location],
    threat_center_lat: float,
    threat_center_lon: float,
    radius_km: float,
) -> float:
    """Fraction of assigned assets placed OUTSIDE the A2/AD contested zone.

    Assets inside radius_km are at high risk; those outside contribute to
    survivable posture. Returns a value in [0, 1]; higher is safer.
    """
    from .core import haversine_distance
    if not assignment:
        return 0.0
    loc_map = {loc.location_id: loc for loc in locations}
    outside = sum(
        1 for lid in assignment.values()
        if lid in loc_map
        and haversine_distance(
            threat_center_lat, threat_center_lon,
            loc_map[lid].lat, loc_map[lid].lon,
        ) > radius_km
    )
    return outside / len(assignment)
