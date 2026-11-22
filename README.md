# Decode the Bioreactor

Predicting cell growth, substrate concentration, and byproduct formation from initial conditions and bioreactor operating data.

## 1. The Challenge

Given initial measurements and a known bioreactor operating trajectory, predict four quantities over time:

- **VCD** — Viable Cell Density
- **TCD** — Total Cell Density
- **Substrate** — Nutrient concentration
- **Byproduct** — Metabolic byproduct concentration

Our dataset contains 49 labeled training batches and 12 hidden test batches. Model performance is evaluated using Mean Absolute Percentage Error (MAPE).

## 2. Exploratory Data Analysis

### 2.1. What does a typical behaviour run look like?

Here are the four quantities we need to predict - plotting all known batches and the mean trajectory. Note that they are all densities, i.e., quantity per volume:

![mean_trajectories_and_overall_behavious](readme_pics/mean_trajectories_measures.png)

Note that the batch variability is relatively small early on and becomes more pronounced later, especially for cell densities. This suggests that differences between batches become more important as the process progresses.

And here are the 4 feeds over time, the color indicated the amount:

![feeds_over_time](readme_pics/feeds_amount_per_time.png)


### 2.2. Does increasing volume explain declining cell density?

![cells_and_dilution](readme_pics/cells_and_dilution.png)

Comparing VCD and TCD with their volume-adjusted equivalents (VCD × volume and TCD × volume).

The volume steadily increases, which suggests that dilution can explain at least part of the decrease of VCD and TCD. (But does it explain everything?)


### 2.3. Where are the difficult prediction regions?

![cv_error_heatmap](readme_pics/cv_error_heatmap.png)

In this version, errors in Substrate and Byproduct increase over time, while the VCD and TCD are more stable.

## 3. Insights and Ideas

### 3.1. Reactor operations reveal biological information

The substrate feeding schedule appears to encode information about the underlying substrate concentration. By examining the relationship between measured substrate levels and subsequent feeding decisions, we identified two consistent controller rules:

Days 7–9: Substrate feeding is triggered when the concentration falls below approximately 1.95 (112/112 checked decisions consistent).

Days 11–12: A higher feed mass is administered when the concentration falls below approximately 0.965 (78/78 checked decisions consistent).

Feed events consistently follow the corresponding measurements by approximately 0.51–0.56 elapsed-time units.

These observations effectively turn the feeding schedule into an indirect substrate sensor. We incorporate the inferred threshold constraints into our predictions, allowing us to recover information about substrate concentrations even when the measurements themselves are hidden.

**[Illustration: Substrate trajectory, controller thresholds, and corresponding feed events]**

### 3.2. Operating signals can reveal batch-specific behavior

Feed schedule deviations, unusual feeding events, and changes in volume provide useful information beyond initial conditions.

In particular, volume changes not explained by the raw feed measurements can act as empirical proxies for process activity, although feed mass-to-volume conversion is not explicitly known.

**[Illustration: Recorded volume changes versus feed-derived proxy]**

### 3.3. Time points are related — especially with limited training data

Rather than fitting 13 completely independent regressions, we use temporally regularized models that encourage neighboring prediction steps to share information.

This reduces sensitivity to noise while allowing the process dynamics to change over time.

**[Illustration: Independent Ridge versus fused Ridge coefficient trajectories]**

### 3.4. Different targets require different modeling strategies

Cell densities, substrate, and byproduct respond differently to operating conditions.

We therefore combine target-specific feature engineering with suitable regression methods, including robust fused Ridge, gradient boosting, and RBF-SVR.

For substrate, we additionally condition predictions on inferred controller constraints.

**[Illustration: Simplified model architecture]**

## 4. Modeling Approach

Our final approach combines:

- Initial conditions and engineered feeding/volume features.
- Temporally regularized and robust regression.
- Nonlinear regression for complementary predictions.
- Controller-informed substrate inference.
- Target-specific model blending.

**[Figure 5: Model pipeline diagram]**

## 5. Results

**[Table: Cross-validation MAPE by target, plus overall MAPE]**

**[Figure 6: Actual versus predicted trajectories for selected validation batches]**

Briefly discuss model strengths, remaining difficulties, and performance on the hidden test set, if available.

## 6. Limitations and Lessons Learned

- Small dataset: only 49 independent labeled batches.
- Some mechanistic quantities are available only through normalized measurements or empirical proxies.
- Late-stage metabolite predictions remain challenging.
- Repeated model selection on the same cross-validation data can lead to optimistic performance estimates.

## 7. How to Run

**[Instructions for environment setup, model execution, cross-validation, and generating submission files]**

## 8. Team and Acknowledgments
Team members:
Gülce Berfin Ercan, Hümay Zehra Özer, Svetlana Tsizin, Karina Samvelyan.
