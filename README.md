# Physics-Informed Machine Learning for Short-Term Aftershock Forecasting

> **Physics-Informed Machine Learning for Short-Term Aftershock Forecasting: A Comparative Study with ETAS**  
> *Agnish Brahma, Avrajit Kundu, Kajal Kumari*  
> Institute of Engineering and Management, Kolkata, India

## Overview

Short-term aftershock forecasting is critical for disaster response, but traditional models like the Epidemic-Type Aftershock Sequence (ETAS) are computationally intensive and parameter‑sensitive. Machine learning offers flexibility but can be brittle when trained on noisy or biased data.

This study introduces a **physics‑informed feature** – the *Boussinesq Stress Index* – derived from seismic magnitude and depth. Using a realistically simulated ETAS catalog, we compare:

- Random Forest (baseline: magnitude + depth)
- Random Forest (enhanced with stress index)
- Logistic Regression (enhanced with stress index)
- ETAS “oracle” (true probabilistic upper bound)

The forecasting task is a **binary decision**: will an earthquake be followed by a larger event (ΔM ≥ 1.0) within 7 days and 50 km?

## Key Findings

| Model | AUC‑ROC | Brier Score | PR‑AUC | F1 |
|-------|---------|-------------|--------|----|
| RF (mag+depth) | 0.684 | 0.146 | 0.313 | 0.793 |
| RF (+stress index) | 0.685 | 0.146 | 0.327 | 0.787 |
| **LR (+stress index)** | **0.744** | 0.205 | **0.389** | 0.726 |
| ETAS Oracle | 0.730 | 0.165 | 0.368 | 0.758 |

- Adding the stress index **does not significantly improve** Random Forest (ΔAUC = +0.0012, 95% CI includes zero).
- Logistic regression **benefits substantially** from the stress index, outperforming both RF variants and the ETAS oracle in discrimination (AUC 0.744).
- SHAP analysis confirms that the stress index is the **second most important feature** in the RF model, yet this importance does not translate into predictive gain – a key negative result.

> **Takeaway:** The value of a physics‑informed feature depends critically on the choice of machine learning model. A simple linear model can sometimes exploit such features more effectively than complex ensembles.

## Methodology

### Data Simulation
- Space‑time ETAS model over 5 years, 500×500 km² domain
- Magnitude completeness \( M_c = 3.0 \), \( b = 1.0 \)
- Adaptive calibration of productivity \( K_0 \) to achieve 15–35% positive cases
- Final catalog: **4,724 events**, 19.4% positive (larger follower)

### Physics‑Informed Feature – Boussinesq Stress Index

- High for shallow, large earthquakes – those most likely to trigger larger aftershocks
- Cheap to compute, ignores shear stress and fault orientation but captures first‑order physics

### Models & Training
- Train/test split: first 80% → training (temporal order preserved)
- Balanced class weights to handle imbalance
- Robust scaling (median + IQR) – fit only on training data
- RF: 100 trees; LR: L2 regularization, 1000 iterations max

### Evaluation Metrics
- AUC‑ROC (discrimination)
- Brier score (calibration)
- Precision‑Recall AUC (imbalance‑robust)
- Reliability diagrams, bootstrap CIs for AUC differences

## Results Highlights

- The physics‑enhanced LR achieves the **highest discrimination** (AUC 0.744) but poorer calibration (Brier 0.205) compared to the oracle.
- Reliability curves show that **all models are miscalibrated** to some degree – a common challenge in probabilistic forecasting.
- SHAP (Fig. 4 in paper) ranks stress index **second only to depth** in importance for the RF model.
- Prediction uncertainty for RF (std dev across trees) averages 0.189, indicating moderate ensemble variability.

## Limitations

- **Simulated data** – lacks real‑world complexities (fault heterogeneity, 3D structure, foreshocks)
- **Direct‑only oracle** – ignores cascading triggers, slightly underestimates true probabilities
- **Single physics feature** – other measures (e.g., Coulomb stress change) may yield different results
- **Limited model set** – no gradient boosting, neural networks, or Bayesian methods
- **Fixed spatial domain** – 500×500 km² region may not generalize globally

## Future Work

- Validate on real catalogs (Japan, California) after declustering
- Incorporate 3D Coulomb stress changes using fault plane solutions
- Hybrid models that use ETAS probabilities as features or correct residuals
- Temporal cross‑validation with expanding time windows
- Deeper interpretability (SHAP/LIME) to understand why LR benefits more than RF
- Ensemble uncertainty quantification for well‑calibrated prediction intervals
