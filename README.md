# Project Overview
This project represents the early stage of a machine learning system designed to predict football match outcomes. The model predicts match results (**Home Win / Draw / Away Win**) using historical match data, including team performance statistics, Elo ratings, expected goals (xG), and engineered features.

The long-term objective is to simulate and predict outcomes of the **2026 FIFA World Cup**.

---

## Current Progress
At this stage, the project includes:

- Data cleaning and preprocessing
- Initial feature engineering
- Training and comparison of multiple classification models
- Evaluation using accuracy, F1-score, and confusion matrices
- Saving the best-performing model for future development

> This version establishes a **baseline predictive framework** rather than a final system.

---

## Limitations & Known Issues
This early version has several known limitations:

- Feature engineering is limited and requires significant expansion
- Certain features (e.g., Elo ratings) may be incorrectly represented or scaled
- The dataset lacks contextual factors such as form, injuries, venue effects, and tournament stage
- Model variance remains high, indicating potential overfitting
- The current simulation results should be interpreted as **experimental**, not definitive predictions

---

## Planned Improvements

- Meaningful feature engineering
- Correction of flawed feature representations
- Integration of additional datasets
- Experimentation with advanced models (ensembles, calibration models, neural networks)
- A more realistic and interpretable **World Cup simulation engine**
