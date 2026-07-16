# Research: Posture and Sustainment Optimization

**Anote, Inc. | Defense Analytics Research**

We propose a greedy placement and reinforcement learning framework for optimizing joint force sustainment scheduling across distributed theater locations, maximizing readiness coverage while minimizing logistics cost under stochastic asset degradation.

---

## Problem Context (DASH-5)

Modern joint force sustainment planning requires optimizing the placement and replenishment of military assets across distributed theater locations under uncertainty. The **DASH-5** problem involves scheduling maintenance, resupply, and repositioning actions to maintain readiness while minimizing logistics cost across M assets, N locations, and T planning horizons.

---

## Problem Formulation

Given:
- **M assets** (aircraft, fuel depots, maintenance crews, munitions, medical)
- **N locations** (bases/FOBs with strategic value and capacity constraints)
- **T time steps** (degradation + replenishment horizon)

Optimize:
- Asset placement assignments (M x N)
- Sustainment action schedule (M x T): REPOSITION, RESUPPLY, MAINTAIN, HOLD

Subject to:
- Capacity constraints per location
- Readiness floor requirements
- Cost budget

---

## RL Framing

The problem maps naturally to a Markov Decision Process:
- **State**: readiness rates, quantities, maintenance timers per asset
- **Action**: sustainment decision per asset per time step
- **Reward**: readiness coverage - cost
- **Transition**: stochastic degradation of readiness/maintenance timers

---

## Optimization Approach

1. **Greedy baseline**: Assign assets to highest-value locations with capacity; apply rule-based ReplenishmentPolicy
2. **Simulation**: Stochastic degradation + policy rollout over T steps
3. **Metrics**: Readiness score, coverage score, sustainment cost, posture efficiency

---

## Metrics Table

| Metric | Definition |
|---|---|
| Readiness Score | Quantity-weighted mean readiness rate |
| Coverage Score | Fraction of locations with ≥1 asset |
| Sustainment Cost | Sum of action costs (REPOSITION=10, RESUPPLY=5, MAINTAIN=2, HOLD=0) |
| Posture Efficiency | (readiness × coverage) / log(2 + cost) |

---

## Python Package (`src/postureopt/`)

### Install

```bash
pip install -e ".[dev]"
```

### Quick Start

```python
from postureopt.data import make_posture_state, simulate_degradation
from postureopt.evaluate import simulation_summary

state = make_posture_state(n_assets=20)
history = simulate_degradation(state, n_steps=10)
print(simulation_summary(history))
```

### Run Tests

```bash
pytest tests/ -v --cov=src
```

---

## Venues

This research is suitable for reframing toward:
- **DAI 2026** (Defense AI)
- **MILCOM** (sustainment scheduling track)
- **NeurIPS** (RL for logistics)

---

## Government Disclaimer

This research is conducted independently by Anote, Inc. Any resemblance to specific classified programs is coincidental. No export-controlled data is used.

---

## Citation

```bibtex
@techreport{anote_postureopt2026,
  author = {Vidra, Natan and Anote, Inc.},
  title  = {Posture and Sustainment Optimization via Greedy Placement and Reinforcement Learning},
  year   = {2026},
  url    = {https://github.com/anote-ai/research-postureandsustainmentoptimization}
}
```
