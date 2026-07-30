# Telco-Customer-Churn-Prediction-Model-
Machine learning project to predict customer churn using the Telco Customer Churn dataset with data preprocessing, feature engineering, EDA, and predictive modeling.

## Project Overview

This project aims to predict whether a telecommunications customer is likely to churn using customer demographics, account details, subscribed services, and billing information. The project follows a complete machine learning workflow, including data cleaning, feature engineering, exploratory data analysis, model training, and evaluation.

## Problem Statement

Customer churn has a significant impact on business revenue. The objective of this project is to identify customers who are likely to leave the company and provide insights that can help improve customer retention strategies.


## Data Quality Issues

The following issues were identified during data exploration:

- TotalCharges contained missing values and an incorrect data type.
- Missing values were handled using the median value, **fit on the training split only** and applied to validation/test (`src/data_processing.py::fill_missing_numeric`) — the earlier version filled with a dataset-wide median before splitting, a small leak.
- Customers with zero tenure required special handling during feature engineering (`AvgMonthlySpend` guards divide-by-zero for `tenure == 0`).
- `TenureGroup`'s bin edges originally left customers at exactly `tenure == 0` **and** `tenure == 72` (min and max of the observed range) as `NaN`; the bins were corrected to cover the full range.
- `customerID` was accidentally one-hot encoded in an early version of the notebook, adding ~7,000 near-unique dummy columns to the feature matrix (one per customer). It's an identifier, not a feature, and is now always dropped before encoding (`src/data_processing.py::prepare_features`).
- Duplicate records were checked before model development.
- Categorical variables required encoding before model training.


## Data Preprocessing

- Converted TotalCharges to numeric.
- Handled missing values.
- Encoded categorical variables.
- Performed train-test split.
- Standardized data preparation using code to ensure reproducibility.


## Feature Engineering

Five new features were created:

- TenureGroup
- AvgMonthlySpend
- NumServices
- IsLongTermContract
- HasFamilySupport

These features were designed to capture customer loyalty, spending behavior, service usage, contract commitment, and household characteristics.


## Exploratory Data Analysis

The analysis showed several important trends:

- Customers on month-to-month contracts churn more frequently.
- Customers with shorter tenure have a higher churn rate.
- Higher monthly charges are associated with increased churn.
- Customers subscribed to more services generally have lower churn.
- Long-term contracts improve customer retention.


## Model Performance

Three models were compared on the **same** preprocessing, the **same**
stratified train/val/test split, and the **same** evaluation harness
(precision, recall, F1, ROC-AUC, confusion matrix — accuracy alone isn't
trusted here because churn is imbalanced, ~1 in 4 customers). Full detail,
including confusion-matrix and ROC plots, is in
`reports/churn_model_comparison.md`; reproduce it with `python -m src.torch_trainer`.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.746 | 0.515 | 0.801 | 0.627 | 0.840 |
| Random Forest | 0.780 | 0.603 | 0.498 | 0.546 | 0.818 |
| Neural Network (MLP) | 0.752 | 0.525 | 0.705 | 0.602 | 0.827 |

*(The neural net's exact numbers vary by a few hundredths across reruns — CPU floating-point ops aren't bit-exact reproducible run to run even with a fixed seed — but Logistic Regression won on F1 in every rerun during development.)*

**Is the neural network worth it here? No — Logistic Regression wins on F1
and ROC-AUC**, and it's simpler, faster to train, and easier to explain to a
business stakeholder. See `reports/churn_model_comparison.md` for the full
writeup of why. (An earlier version of this table reported 1.00 accuracy for
Logistic Regression — that was a copy-paste error that never matched the
notebook's own printed output. It's been replaced with the real, regenerated
numbers above.)

## Deep Learning Extension (PyTorch)

This project also includes a from-scratch-to-framework PyTorch exercise built
on top of the churn work above:

- **Part 1 — Tensors & autograd** (`notebooks/pytorch_fundamentals.ipynb`):
  tensor shapes, hand-derived vs. autograd gradients, and a from-scratch
  NumPy backward pass checked numerically against PyTorch autograd on the
  same 2→8→1 network (they match to floating-point precision).
- **Part 2 — Training loop practice** (same notebook): a full train/val loop
  on `make_moons`, plus a one-lever-at-a-time table (learning rate,
  width/depth, epochs, dropout, weight decay, input normalization) showing
  what each change actually did to validation loss.
- **Part 3 — The churn network** (`src/torch_trainer.py`): a small MLP
  (`input → 64 → 32 → 1`, ReLU, dropout) trained on the exact same
  preprocessing, split, and evaluation harness as the classical models
  above, with `pos_weight` and threshold-tuning to handle the ~1-in-4 class
  imbalance. Results and the honest "was it worth it" writeup are in
  `reports/churn_model_comparison.md`.

## Limitations

- The dataset represents a single telecommunications company.
- No external customer behavior data was available.
- Hyperparameter tuning was not extensively performed.
- Results may vary with different datasets.



## Recommendations

1. Encourage customers on month-to-month contracts to switch to longer-term plans.
2. Develop retention campaigns targeting new customers during their first year.
3. Provide personalized offers for high-value customers who show a high risk of churn.


## Project Structure

```
Telco-Customer-Churn-Prediction-Model/
│
├── config/
├── data/
│   ├── raw/                  # original Telco-Customer-Churn.csv
│   ├── processed/            # cleaned + feature-engineered CSVs (regenerable)
│   └── external/
├── notebooks/
│   ├── telco_customer_churn.ipynb    # EDA, cleaning, feature engineering, classical models
│   └── pytorch_fundamentals.ipynb    # Part 1 (tensors/autograd) + Part 2 (make_moons)
├── models/                   # saved RF / LR / scaler / MLP artifacts (regenerable)
├── reports/
│   └── churn_model_comparison.md     # Part 3 deliverable: metrics, plots, writeup
├── src/
│   ├── data_processing.py    # reusable clean/feature-engineer/prepare pipeline
│   └── torch_trainer.py      # Part 3: trains + compares RF, LR, and the MLP
├── tests/
│   └── test_data_processing.py
├── README.md
├── requirements.txt
```

## How to Reproduce

```bash
pip install -r requirements.txt

# Regenerate the processed CSVs from the raw data
python -m src.data_processing

# Train all three models and regenerate reports/churn_model_comparison.md
python -m src.torch_trainer

# Run the test suite
pytest tests/
```

