# ML/AI Approaches

## Supervised signals → Portfolio
Pipeline: train a predictor for returns/directions/ranks (see **Risk Premium** task), then map signals to weights (equal-weighted, score-weighted, risk-budgeting).

**Strengths**: leverages the strongest upstream predictors.  
**Limitations**: **turnover/costs** and signal **stability** become the bottlenecks.

## Reinforcement Learning (direct weight decisions)
- **Return objective (sketch)**:
$$
Q^\pi(s,a) = \mathbb{E}\!\left[\sum_{k=t}^{\infty} \gamma_k\, r_k \,\middle|\, s_t, a_t \right].
$$
- **State**: prices/factors, previous weights, volumes, etc.  
- **Action**: current weights; **Constraints**: costs/impact, risk limits, position bounds.

**Strengths**: directly incorporates costs/constraints into the objective.  
**Limitations**: training/generalization difficulty (distribution shift, sparse rewards); requires off-policy evaluation and strong backtesting protocols.