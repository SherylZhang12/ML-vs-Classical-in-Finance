# ML/AI Approaches

## Objective (Aligned with Classic)
Same task and objective as Classic; only the function class changes:
$$
\hat g = \arg\min_{g \in \mathcal{G}} \sum_{i,t} \big( y_{i,t} - g(x_{i,t-1}) \big)^2.
$$

## Models
- **Regularized linear**: Ridge / LASSO (variable selection, shrinkage).
- **Trees/Ensembles**: Random Forest, XGBoost (nonlinearity, interactions, robust importance).
- **DNN/MLP**: flexible nonlinear mapping; use dropout/weight decay/BN for generalization.
- **Transformer (cross-sectional / short temporal context)**: attention to capture cross-asset and feature interactions.

## Strengths & Limitations
**Strengths**
- Handles **high-dimensional, nonlinear, time-varying** relations.
- Built-in regularization; extensible to **classification** (direction) and **ranking** objectives.

**Limitations**
- **Overfitting** risk → strict rolling splits, early stopping, and ensembling are required.
- **Interpretability pressure** → provide factor exposures, neutralization tests, and stability diagnostics.
- Highly sensitive to data leakage and preprocessing choices.