# Research Design Document: Posture and Sustainment Optimization

## Vision Statement

Deliver the first rigorous comparison of **robust optimization, stochastic programming, and reinforcement learning** for military posture and sustainment decisions under deep uncertainty — proving that distributionally robust stochastic programming (DRSP) outperforms standard linear programming by ≥20% on out-of-sample scenarios while remaining computationally tractable for operational planning timelines, and establishing **SustainBench** as the standard evaluation framework for AI-driven posture optimization.

---

## Problem Statement & Novelty

Military posture and sustainment optimization involves deciding: where to pre-position assets, how much stock to hold, which supply routes to use, and how to adapt when plans encounter reality. Current practice relies on:

1. **Deterministic LP models**: Optimize for a single expected scenario, fragile under uncertainty.
2. **Manual scenario analysis**: Planners manually consider 3–5 scenarios; computationally infeasible to explore large uncertainty spaces.
3. **No reinforcement learning comparison**: RL for logistics/sustainment is studied in isolation but never rigorously compared to classical stochastic programming.
4. **No recourse policy evaluation**: Plans must have recourse actions (what to do when the plan fails); existing benchmarks don't evaluate recourse quality.

### Novel Contributions

| Contribution | Description |
|---|---|
| **SustainBench** | 200 posture/sustainment scenarios with parameterized uncertainty, cost functions, and recourse actions |
| **ROE metric** | Robustness-Optimality Efficiency: Pareto score of out-of-sample cost vs. worst-case cost |
| **DRSP framework** | Distributionally robust stochastic programming for posture optimization |
| **RL recourse policy** | Trained RL agent for adaptive recourse decisions |
| **Out-of-sample evaluation protocol** | First systematic OOS evaluation framework for military OR models |

### Key Metrics

```
ROE = (1 / cost_ratio) × robustness_score

where:
  cost_ratio = out_of_sample_cost / optimal_hindsight_cost  (lower is better)
  robustness_score = 1 - worst_case_regret / max_possible_regret  (higher is better)

Out-of-sample evaluation:
  Train on scenario distribution Θ_train
  Evaluate on held-out Θ_test (adversarially chosen)
  Report: mean cost, 95th percentile cost, worst-case cost
```

---

## Research Objectives

1. Demonstrate that **DRSP** achieves ≥20% lower out-of-sample cost vs. deterministic LP across SustainBench scenarios.
2. Show that **SAA (Sample Average Approximation)** is competitive with DRSP at high sample counts but deteriorates faster under distribution shift.
3. Evaluate **RL recourse policies**: do RL agents make better adaptive decisions than rule-based recourse?
4. Characterize the **computation-quality tradeoff**: at what problem size does DRSP become computationally infeasible for operational timelines?
5. Identify **scenario types** where each method excels, providing a decision framework for method selection.

---

## Dataset Construction (SustainBench)

### Scenario Categories (200 scenarios)

| Category | Count | Key Uncertainties | Decision Variables |
|---|---|---|---|
| Humanitarian assistance | 40 | Demand surge, access denial | Pre-positioning, route selection |
| Logistics sustainment | 50 | Supply disruption, transit delays | Stock levels, replenishment timing |
| Force posture (joint) | 40 | Threat activation, force requirements | Asset placement, basing |
| Disaster response | 35 | Disaster magnitude, affected population | Resource allocation, staging |
| Multi-echelon supply | 35 | Lead times, demand variability | Inventory policy, supplier selection |

### Uncertainty Parameterization

```python
# Scenario uncertainty structure
class SustainScenario:
    def __init__(self):
        self.nominal_demand = ...  # Baseline demand vector
        self.uncertainty_set = UncertaintySet(
            type='ellipsoidal',  # or 'box', 'data-driven'
            parameters={'radius': 0.3, 'correlation': 0.6}
        )
        self.recourse_options = [...]  # Available adaptive actions
        self.cost_function = ...  # Multi-objective (cost + readiness + risk)
        self.planning_horizon = 30  # days
        self.oos_test_scenarios = generate_adversarial_scenarios(n=100)
```

### Ground Truth and Evaluation
- **Oracle solution**: Solved with perfect hindsight (lower bound on achievable cost)
- **Out-of-sample test set**: 100 adversarially generated scenarios per category
- **Human expert evaluation**: Operational planners rate solution feasibility and doctrinal compliance

---

## Methods Under Evaluation

| Method | Type | Handles Uncertainty | Recourse | Scalability |
|---|---|---|---|---|
| Deterministic LP | Classical OR | No | None | Excellent |
| SAA (2-stage) | Stochastic | Distributional | Fixed | Good |
| DRSP (ours) | Robust stochastic | Distributional robust | Fixed | Good |
| Robust LP (minimax) | Robust | Worst-case | None | Good |
| RL (PPO) | Reinforcement learning | Implicit | Adaptive | Moderate |
| DRSP + RL recourse | Hybrid | Distributional robust | Adaptive (RL) | Moderate |
| Human expert | Manual | Heuristic | Expert | N/A |

---

## Experimental Design

### Baseline Experiment (Experiment 0)
**Protocol**: Deterministic LP on 40 humanitarian assistance scenarios. Compute mean cost, OOS cost ratio.

**Expected result**: OOS cost ratio ≈ 1.31 (31% above optimal hindsight). Establishes the cost of ignoring uncertainty.

---

### Experiment 1: DRSP vs. SAA vs. LP (Core Comparison)
**Hypothesis**: DRSP achieves ≥20% lower OOS mean cost vs. deterministic LP, and ≥8% lower 95th percentile cost vs. SAA.

**Protocol**:
1. Run LP, SAA, Robust LP, DRSP on all 200 scenarios.
2. Evaluate each solution on OOS test set (100 adversarial scenarios each).
3. Compute: mean OOS cost, 95th percentile OOS cost, worst-case OOS cost, ROE.
4. Statistical test: Wilcoxon signed-rank for DRSP vs. SAA.

**Expected results**:

| Method | Mean OOS Cost Ratio | 95th Pctile Ratio | Worst Case Ratio | ROE |
|---|---|---|---|---|
| Deterministic LP | 1.31 | 1.62 | 2.41 | 0.48 |
| SAA (2-stage) | 1.14 | 1.38 | 1.89 | 0.63 |
| Robust LP | 1.09 | 1.21 | 1.44 | 0.71 |
| DRSP (ours) | 1.07 | 1.18 | 1.39 | 0.76 |

- DRSP improvement over LP: (1.31 - 1.07) / 1.31 = 18% on mean cost, 27% on 95th percentile
- DRSP vs. SAA: better on tail risk (18% lower 95th percentile) while comparable on mean cost

---

### Experiment 2: RL Recourse Policy
**Hypothesis**: RL recourse policy reduces adaptive recourse cost by ≥15% vs. rule-based recourse, at comparable computational overhead.

**Protocol**:
1. Train PPO agent on SustainBench training scenarios (recourse task only).
2. Compare: (a) DRSP + rule-based recourse, (b) DRSP + RL recourse.
3. Measure: recourse cost, constraint feasibility (does RL produce operationally feasible decisions?), training time.

**Expected results**:
- Rule-based recourse cost: index 1.0 (baseline)
- RL recourse cost: 0.84 (−16%)
- Feasibility rate (RL): 91% (vs. 100% rule-based — RL sometimes violates doctrinal constraints)
- Key finding: RL recourse needs constraint-aware training (constrained PPO) to match feasibility of rule-based
- After constrained PPO: cost 0.87 (−13%), feasibility 97%

```python
# Constrained PPO for recourse
class ConstrainedPPORecourse(PPO):
    def compute_reward(self, state, action, next_state):
        base_reward = -self.cost_function(state, action)
        constraint_penalty = sum(
            self.penalty_coeff * max(0, constraint(state, action))
            for constraint in self.doctrinal_constraints
        )
        return base_reward - constraint_penalty
```

---

### Experiment 3: Computation-Quality Tradeoff
**Hypothesis**: DRSP remains solvable within 4-hour planning timeline for scenarios up to 500 decision variables; SAA scales better but with lower OOS quality.

**Protocol**:
1. Vary scenario complexity: 50, 100, 200, 500, 1000 decision variables.
2. Measure solve time for LP, SAA, DRSP at each size.
3. Plot: quality (ROE) vs. solve time for each method.

**Expected results**:

| Problem Size | LP Solve Time | SAA Solve Time | DRSP Solve Time | DRSP ROE |
|---|---|---|---|---|
| 50 vars | 2s | 45s | 3min | 0.78 |
| 100 vars | 8s | 4min | 18min | 0.76 |
| 200 vars | 45s | 20min | 2.1hr | 0.74 |
| 500 vars | 8min | 2.5hr | >12hr | N/A |

- DRSP feasibility limit: ~200 decision variables within 4-hour planning constraint
- For larger problems: SAA is the recommended alternative
- Operational recommendation: DRSP for strategic posture; SAA for tactical sustainment

---

### Experiment 4: Distribution Shift Robustness
**Hypothesis**: DRSP degrades gracefully under distribution shift; SAA degrades sharply when test distribution differs from training distribution by >20% Wasserstein distance.

**Protocol**:
1. Train SAA and DRSP on nominal distributions.
2. Evaluate on increasingly shifted test distributions (measure shift by Wasserstein distance).
3. Compare degradation curves.

**Expected results**:
- At 0 shift: DRSP cost ratio 1.07, SAA 1.14 (SAA better on in-distribution)
- At 20% shift: DRSP 1.12, SAA 1.28 (SAA degrades faster)
- At 40% shift: DRSP 1.18, SAA 1.51
- Key finding: DRSP's robustness guarantee provides 2× better distribution shift resistance than SAA

---

## Expected Results Summary

| Metric | Deterministic LP | DRSP | Improvement |
|---|---|---|---|
| Mean OOS cost ratio | 1.31 | 1.07 | −18% |
| 95th pctile OOS ratio | 1.62 | 1.18 | −27% |
| ROE | 0.48 | 0.76 | +58% |
| Max feasible problem size | Any | 200 vars / 4hr | Scalable for tactical |
| Distribution shift robustness | Low | High | 2× better |

**Primary claim**: Distributionally robust stochastic programming achieves 18–27% lower out-of-sample costs vs. deterministic LP while remaining tractable for scenarios with up to 200 decision variables, making it the recommended approach for strategic military posture optimization.

---

## Why This Matters

**For researchers**: SustainBench provides the first rigorous OR benchmark for military posture optimization with proper OOS evaluation.

**For DoD planners**: A 20% reduction in out-of-sample sustainment costs translates directly to readiness improvements or budget savings at operational scale.

**For OR practitioners**: The computation-quality tradeoff analysis provides concrete guidance for method selection based on problem size and timeline constraints.

---

## Implementation Plan

```
research-postureandsustainmentoptimization/
├── data/
│   ├── scenarios/       # 200 SustainBench scenarios
│   ├── oos_test/        # OOS adversarial test sets
│   └── expert_eval/     # Human expert ratings
├── solvers/
│   ├── lp_baseline.py
│   ├── saa.py
│   ├── drsp.py          # Our DRSP framework
│   ├── robust_lp.py
│   └── rl_recourse.py   # Constrained PPO
├── metrics/
│   ├── roe.py
│   └── oos_eval.py
├── experiments/
│   ├── exp0_baseline.py
│   ├── exp1_core_comparison.py
│   ├── exp2_rl_recourse.py
│   ├── exp3_scalability.py
│   └── exp4_distribution_shift.py
```

---

## Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Scenario construction | 5 weeks | 200 SustainBench scenarios |
| Solver implementation | 4 weeks | All methods implemented |
| Experiments | 5 weeks | All results |
| Expert evaluation | 2 weeks | Feasibility ratings |
| Paper writing | 4 weeks | INFORMS/MORS submission |

**Target venues**: INFORMS Operations Research, MORS Symposium, or Operations Research Letters

---

## Open Questions & Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| DRSP solver convergence | Medium | Commercial solver (Gurobi) + CPLEX fallback |
| RL constraint violations | High | Constrained PPO; projection layer |
| Scenario realism validation | Medium | Expert review panel |
| Computational cost at scale | High | Cloud HPC for large experiments |

---

## Related Issues

- COA Generation: posture decisions feed COA planning
- DARPA LYFT: sustainment is a LYFT-relevant capability
- Reproducibility: solver version pinning
- Related work audit: two-stage stochastic programming, robust optimization literature
