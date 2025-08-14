# Classical Baselines

## Problem & Objective
Choose weights under costs/constraints to maximize risk-adjusted return or investor utility.

## Baselines
### Mean–Variance (MPT)
$$
w^\star = \frac{1}{\gamma}\,\Sigma^{-1}\mu,
$$
where $\mu$ is expected returns, $\Sigma$ is covariance, and $\gamma$ is risk aversion.

### Extensions
- **Minimum-variance**, **Risk Parity**, **Black–Litterman**, **Robust optimization** (shrinkage/regularization).

**Strengths**: closed/semi-closed forms; computationally efficient.  
**Limitations**: highly sensitive to estimation errors in $\hat\mu$ and $\hat\Sigma$; stability issues in high dimensions and under drift.