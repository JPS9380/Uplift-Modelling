# Uplift Modeling with Causal ML

This project demonstrates the application of uplift modeling, a causal inference technique, to estimate the Individual Treatment Effect (ITE) of a marketing treatment on customer conversion. The analysis is performed on the Criteo Uplift Prediction Dataset using the `causalml` library.

---

## Project Overview

The primary goal of this project is to move beyond traditional predictive modeling (which predicts the likelihood of an event) and towards causal inference. Specifically, we want to identify which individuals are most likely to *convert as a result of being targeted by a treatment* (e.g., seeing an advertisement, a discount coupon). This is known as measuring the "uplift."

This is crucial for marketing campaigns to optimize budget allocation by targeting only those customers who will be positively influenced by the treatment (the "persuadables"), and avoiding those who would convert anyway ("sure things"), those who will never convert ("lost causes"), and those who would be deterred by the treatment ("sleeping dogs").

We use a `CausalRandomForestRegressor` to estimate the ITE for each user in the dataset.

--- 

## Dataset

The project uses the [Criteo Uplift Prediction Dataset (v2.1)](https://ailab.criteo.com/criteo-uplift-prediction-dataset/). The original dataset contains approximately 14 million records and is highly imbalanced in favor of the non-conversion class.

To handle this class imbalance and create a more manageable dataset for training, a preprocessing step was performed:
1.  All records where `conversion == 1` (the minority class) were kept (40,774 samples).
2.  A random sample of 1,00,000 records where `conversion == 0` (the majority class) was selected.
3.  These two sets were combined to create a new, balanced sample dataset named `final_sample.csv`.

The dataset includes the following key columns:
- `f0` - `f11`: 12 anonymized user features.
- `treatment`: A binary indicator (1 if the user was in the treatment group, 0 for the control group).
- `conversion`: A binary outcome (1 if the user converted, 0 otherwise).
- `visit`: A binary outcome (1 if the user visited the advertiser's website, 0 otherwise).
- `exposure`: A binary indicator of whether the user was exposed to the treatment.

--- 

## Methodology

The project workflow is divided into two main parts: preprocessing and training.

### 1. Preprocessing (`preprocessing.ipynb`)
- The original `criteo-uplift-v2.1.csv` dataset is loaded.
- The dataset is downsampled to balance the `conversion` classes as described above.
- The resulting balanced dataframe is saved to `dataset/final_sample.csv`.

### 2. Model Training and Inference (`training.ipynb`)
- The preprocessed dataset `final_sample.csv` is loaded.
- The data is prepared for the model by separating it into features (X), a treatment indicator (W), and an outcome (Y).
  - **Features (X)**: `f0` through `f11`.
  - **Treatment (W)**: `treatment`.
  - **Outcome (Y)**: `conversion`.
- A `CausalRandomForestRegressor` model from the `causalml` library is instantiated with 200 estimators.
- The model is trained on the data to learn the relationship between the features, treatment, and outcome.
- The trained model is used to predict the **Individual Treatment Effect (ITE)** for each user. The ITE represents the predicted change in conversion probability for a user if they receive the treatment versus if they do not.
- The results, including the original treatment/conversion status and the predicted ITE, are stored in a final DataFrame.

---

## File Structure
```
├── dataset/
│   ├── criteo-uplift-v2.1.csv   # Original dataset (must be downloaded)
│   └── final_sample.csv         # Preprocessed, sampled dataset
├── model/
│   └── model.joblib             # Saved trained model
├── analysis.py                  # python script to understand uplift modelling better(getting the number of users
                                 # who have more than 5% increment in conversion rate if provided with treatment)
├── preprocessing.ipynb          # Jupyter notebook for data preprocessing
└── training.ipynb               # Jupyter notebook for model training and evaluation
```

---