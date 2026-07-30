"""
Data processing utilities for churn prediction.

Mirrors the cleaning / feature-engineering steps developed interactively in
``notebooks/telco_customer_churn.ipynb`` so they can be reused from scripts
(``src/torch_trainer.py``) and tests instead of copy-pasted.

Design choice — no leakage:
    ``clean_data`` only fixes dtypes; it does NOT impute missing values using
    dataset-wide statistics. Imputation (e.g. filling TotalCharges with a
    median) must be fit on the training split only, then applied to
    validation/test. See ``fill_missing_numeric`` below, and how
    ``src/torch_trainer.py`` calls it after the train/val/test split.

Known bug fixed here (present in the original notebook):
    The notebook one-hot encoded ``customerID`` before modelling, which
    silently added ~7,000 near-unique dummy columns (one per customer) to
    the feature matrix. ``customerID`` is an identifier, not a feature, and
    is always dropped before encoding in this module.
"""

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_data(path: str) -> pd.DataFrame:
    """Load raw data from CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes and drop exact duplicate rows. No imputation happens here
    (see module docstring) — missing values are left as NaN on purpose.
    """
    df = df.copy()

    # TotalCharges arrives as a string (blank for a handful of tenure==0
    # customers); coerce to numeric, turning those blanks into NaN.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.drop_duplicates()
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the five engineered features used throughout the project."""
    df = df.copy()

    # TenureGroup: bins span [-1, 72] (inclusive on the right, as pd.cut
    # defaults to) so both ends of the observed tenure range (0 and 72) land
    # in a real bucket instead of NaN. The notebook's original bins,
    # [0, 12, 24, 48, 72] with the default right-closed behaviour, silently
    # dropped every tenure == 0 customer (new signups) into a null bucket.
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New", "Regular", "Loyal", "Very Loyal"],
    )

    # AvgMonthlySpend: guard divide-by-zero for tenure == 0 customers.
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    df["NumServices"] = (
        df[SERVICE_COLUMNS]
        .replace(
            {
                "Yes": 1,
                "No": 0,
                "No internet service": 0,
                "No phone service": 0,
            }
        )
        .sum(axis=1)
    )

    df["IsLongTermContract"] = df["Contract"].isin(["One year", "Two year"]).astype(int)

    df["HasFamilySupport"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)

    return df


def build_processed_frame(raw_path: str) -> pd.DataFrame:
    """Full pipeline: raw CSV -> cleaned + feature-engineered DataFrame.

    Equivalent to what the notebook produces, but reusable as a function and
    without the customerID one-hot bug.
    """
    df = load_data(raw_path)
    df = clean_data(df)
    df = engineer_features(df)
    return df


def fill_missing_numeric(
    train: pd.DataFrame,
    columns: Iterable[str],
    *others: pd.DataFrame,
) -> tuple:
    """Fit medians on ``train`` only, then fill NaNs in ``train`` and every
    frame in ``others`` (e.g. validation, test) with those same values.

    Returns a tuple of the filled frames in the same order they were passed
    in: ``(train, *others)``.
    """
    columns = list(columns)
    medians = train[columns].median()

    train = train.copy()
    train[columns] = train[columns].fillna(medians)

    filled_others = []
    for frame in others:
        frame = frame.copy()
        frame[columns] = frame[columns].fillna(medians)
        filled_others.append(frame)

    return (train, *filled_others)


def prepare_features(
    df: pd.DataFrame,
    drop_columns: Optional[Iterable[str]] = None,
) -> tuple:
    """Split a processed frame into a model-ready (X, y).

    Drops the identifier and target column, one-hot encodes the remaining
    categoricals. ``customerID`` is always excluded — see module docstring.
    """
    df = df.copy()
    y = (df[TARGET_COLUMN] == "Yes").astype(int)

    drop = set(drop_columns or [])
    drop.update({ID_COLUMN, TARGET_COLUMN})
    drop = [c for c in drop if c in df.columns]

    X = df.drop(columns=drop)
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def main():
    """Regenerate the processed CSVs under data/processed/ from the raw CSV."""
    try:
        root = Path(__file__).resolve().parents[1]
    except NameError:
        root = Path.cwd()

    raw_path = root / "data" / "raw" / "Telco-Customer-Churn.csv"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = load_data(str(raw_path))
    cleaned = clean_data(raw)
    cleaned.to_csv(processed_dir / "telco_customer_churn_clean.csv", index=False)

    featured = engineer_features(cleaned)
    featured.to_csv(processed_dir / "telco_customer_churn_features.csv", index=False)

    print(f"Wrote {processed_dir / 'telco_customer_churn_clean.csv'} ({cleaned.shape})")
    print(f"Wrote {processed_dir / 'telco_customer_churn_features.csv'} ({featured.shape})")


if __name__ == "__main__":
    main()
