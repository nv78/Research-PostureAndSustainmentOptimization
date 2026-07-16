"""Adversarial Bayesian counter-move model and robust CEV optimizer (Experiment 3).
MinimaxOptimizer added for Experiment 5 cross-baseline comparison.

Builds on core.ThreatScenario (threat_levels / weight) and
core.ScenarioWeightedOptimizer rather than duplicating them.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .core import Asset, Location, ScenarioWeightedOptimizer, ThreatScenario


class AdversarialModel:
    """Bayesian counter-move model for an A2/AD adversary.

    The adversary observes the defender's placement with probability p_obs and,
    if rational, shifts scenario weights toward those whose threat patterns most
    damage the observed placement (maximising expected disruption).

    p_obs: float in [0, 1] — observation probability.
    rationality: float in [0, 1] — interpolates random (0) to fully Bayesian (1).
    blend = p_obs * rationality controls the strength of the adversarial update.
    """

    def __init__(self, p_obs: float, rationality: float = 1.0) -> None:
        if not (0.0 <= p_obs <= 1.0):
            raise ValueError(f"p_obs must be in [0, 1], got {p_obs}")
        if not (0.0 <= rationality <= 1.0):
            raise ValueError(f"rationality must be in [0, 1], got {rationality}")
        self.p_obs = p_obs
        self.rationality = rationality

    def update_weights(
        self,
        scenarios: List[ThreatScenario],
        assignment: Dict[str, str],
        locations: List[Location],
    ) -> List[ThreatScenario]:
        """Return scenarios with weights shifted toward adversarially-preferred ones.

        Adversarial score of scenario s = sum_l(threat_level_s(l) * assets_at(l)).
        With blend = p_obs * rationality we interpolate each scenario's weight from
        its original value toward the adversarially-proportional weight.
        Returns the original list unchanged when blend=0 or no adversarial signal.
        """
        blend = self.p_obs * self.rationality
        if blend == 0.0:
            return scenarios

        location_counts: Dict[str, int] = {loc.location_id: 0 for loc in locations}
        for loc_id in assignment.values():
            if loc_id in location_counts:
                location_counts[loc_id] += 1

        adv_scores = [
            sum(
                s.threat_levels.get(lid, 0.0) * count
                for lid, count in location_counts.items()
            )
            for s in scenarios
        ]
        total_score = sum(adv_scores)
        if total_score == 0.0:
            return scenarios

        total_weight = sum(s.weight for s in scenarios)
        adv_weights = [sc / total_score * total_weight for sc in adv_scores]

        return [
            ThreatScenario(
                scenario_id=s.scenario_id,
                threat_levels=s.threat_levels,
                weight=(1.0 - blend) * s.weight + blend * adv_w,
            )
            for s, adv_w in zip(scenarios, adv_weights)
        ]


class MinimaxOptimizer:
    """Minimax (worst-case) placement optimizer.

    Ranks locations by the minimum effective strategic value across all scenarios.
    Places greedily on this minimax ranking, maximising the guaranteed value
    under the single worst scenario in the set.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def optimize_placement(
        self,
        assets: List[Asset],
        locations: List[Location],
        scenarios: List[ThreatScenario],
    ) -> Dict[str, str]:
        """Assign assets greedily by worst-case effective strategic value."""
        def minimax_value(loc: Location) -> float:
            if not scenarios:
                return loc.strategic_value
            return min(
                loc.strategic_value * (1.0 - s.threat_levels.get(loc.location_id, 0.0))
                for s in scenarios
            )

        sorted_locs = sorted(locations, key=lambda loc: -minimax_value(loc))
        capacity_remaining = {loc.location_id: loc.capacity for loc in sorted_locs}
        assignment: Dict[str, str] = {}
        for asset in assets:
            for loc in sorted_locs:
                if capacity_remaining[loc.location_id] > 0:
                    assignment[asset.asset_id] = loc.location_id
                    capacity_remaining[loc.location_id] -= 1
                    break
        return assignment


class RobustCEVOptimizer:
    """Adversarially-robust CEV optimizer (Experiment 3).

    Iterates: place via ScenarioWeightedOptimizer → adversary shifts weights →
    re-place, until the placement stabilises (or max_iter). Returns the converged
    placement and the final (adversarially-updated) scenario weights.
    """

    def __init__(
        self,
        adversary: AdversarialModel,
        seed: int = 42,
        max_iter: int = 10,
    ) -> None:
        self.adversary = adversary
        self.seed = seed
        self.max_iter = max_iter

    def optimize_placement(
        self,
        assets: List[Asset],
        locations: List[Location],
        scenarios: List[ThreatScenario],
    ) -> Tuple[Dict[str, str], List[ThreatScenario]]:
        """Return (placement, final_scenarios) at adversarial convergence."""
        current = scenarios
        assignment: Dict[str, str] = {}
        cev = ScenarioWeightedOptimizer(seed=self.seed)
        for _ in range(self.max_iter):
            new_assignment = cev.optimize_placement(assets, locations, current)
            updated = self.adversary.update_weights(current, new_assignment, locations)
            if new_assignment == assignment:
                break
            assignment = new_assignment
            current = updated
        return assignment, current
