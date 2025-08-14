#  Classical Baselines

## Problem & Objective
**Task**: cross-sectional prediction of excess returns (regression, direction, or ranking).  
**Objective**:
$$
y_{i,t} = g(x_{i,t-1}) + \varepsilon_{i,t}, \quad \text{with classic } g(\cdot) \text{ being linear/additive.}
$$

## Models

### Linear factor / cross-sectional linear model
$$
y_{i,t} = \alpha_{i,t-1} + \beta_{i,t-1}' f_t + \varepsilon_{i,t}.
$$
- **Interpretation**: a few common factors $f_t$ explain cross-sectional returns; under no mispricing $\alpha \approx 0$.
- **Usage**: baseline for **explainability** (alphas, loadings stability, characteristic-sorted portfolios).

### Latent factors from return panels (IPCA-style idea)
$$
y_{i,t} = x_{i,t-1}\,\beta_i\,f_t + \varepsilon_{i,t}.
$$
- **Idea**: characteristics drive **time-varying loadings/factors**, performing dimensionality reduction on the return panel.
- **Caveat**: high-dimensional parameterization → estimation efficiency and misspecification sensitivity.

## Strengths & Limitations
**Strengths**
- Clear economic interpretation (factors and exposures); established alpha and portfolio diagnostics.
- Simple and fast; easy to run robustness checks.

**Limitations**
- **Linearity/additivity** constraints; limited capacity for **nonlinearity** and **time variation**.
- Potential **parameter explosion** in high dimensions.