import pandas as pd
import pytest

from src.data_processing import (
    clean_data,
    engineer_features,
    fill_missing_numeric,
    prepare_features,
)


@pytest.fixture
def raw_frame():
    return pd.DataFrame(
        {
            "customerID": ["0001-AAA", "0002-BBB", "0003-CCC", "0004-DDD"],
            "gender": ["Female", "Male", "Male", "Female"],
            "tenure": [0, 5, 24, 72],
            "TotalCharges": [" ", "100.5", "500.0", "2000.0"],
            "MonthlyCharges": [20.0, 20.1, 25.0, 30.0],
            "PhoneService": ["Yes", "Yes", "No", "Yes"],
            "MultipleLines": ["No phone service", "No", "No", "Yes"],
            "OnlineSecurity": ["No", "Yes", "No internet service", "Yes"],
            "OnlineBackup": ["No", "No", "No internet service", "Yes"],
            "DeviceProtection": ["No", "No", "No internet service", "Yes"],
            "TechSupport": ["No", "No", "No internet service", "Yes"],
            "StreamingTV": ["No", "No", "No internet service", "Yes"],
            "StreamingMovies": ["No", "No", "No internet service", "Yes"],
            "Contract": ["Month-to-month", "Month-to-month", "Two year", "One year"],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "No", "Yes"],
            "Churn": ["No", "Yes", "No", "No"],
        }
    )


def test_clean_data_coerces_total_charges_and_leaves_nan(raw_frame):
    cleaned = clean_data(raw_frame)
    assert cleaned["TotalCharges"].dtype.kind == "f"
    # the blank string for the tenure==0 customer becomes NaN, not 0 or dropped
    assert cleaned["TotalCharges"].isna().sum() == 1


def test_clean_data_drops_exact_duplicates(raw_frame):
    dup = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)
    cleaned = clean_data(dup)
    assert len(cleaned) == len(raw_frame)


def test_engineer_features_tenure_zero_is_not_null(raw_frame):
    cleaned = clean_data(raw_frame)
    featured = engineer_features(cleaned)
    # regression test for the notebook bug: tenure == 0 must land in "New",
    # not fall through pd.cut's bins as NaN
    assert featured.loc[featured["tenure"] == 0, "TenureGroup"].iloc[0] == "New"


def test_engineer_features_max_tenure_is_not_null(raw_frame):
    cleaned = clean_data(raw_frame)
    featured = engineer_features(cleaned)
    assert featured.loc[featured["tenure"] == 72, "TenureGroup"].iloc[0] == "Very Loyal"


def test_engineer_features_num_services_counts_yes_only(raw_frame):
    cleaned = clean_data(raw_frame)
    featured = engineer_features(cleaned)
    # customer 0003-CCC: PhoneService=No, everything else "No internet service"
    assert featured.loc[featured["customerID"] == "0003-CCC", "NumServices"].iloc[0] == 0
    # customer 0004-DDD: all eight service columns are "Yes"
    assert featured.loc[featured["customerID"] == "0004-DDD", "NumServices"].iloc[0] == 8


def test_engineer_features_is_long_term_contract(raw_frame):
    cleaned = clean_data(raw_frame)
    featured = engineer_features(cleaned)
    flags = dict(zip(featured["customerID"], featured["IsLongTermContract"]))
    assert flags["0001-AAA"] == 0  # Month-to-month
    assert flags["0003-CCC"] == 1  # Two year
    assert flags["0004-DDD"] == 1  # One year


def test_engineer_features_has_family_support(raw_frame):
    cleaned = clean_data(raw_frame)
    featured = engineer_features(cleaned)
    flags = dict(zip(featured["customerID"], featured["HasFamilySupport"]))
    assert flags["0001-AAA"] == 1  # Partner == Yes
    assert flags["0002-BBB"] == 0  # neither
    assert flags["0004-DDD"] == 1  # both


def test_avg_monthly_spend_no_divide_by_zero(raw_frame):
    cleaned = clean_data(raw_frame)
    featured = engineer_features(cleaned)
    # tenure == 0 must not raise / produce inf; TotalCharges is NaN there anyway
    assert not (featured["AvgMonthlySpend"].replace([float("inf"), float("-inf")], pd.NA).eq(float("inf"))).any()


def test_prepare_features_drops_id_and_target(raw_frame):
    cleaned = clean_data(raw_frame)
    featured = engineer_features(cleaned)
    X, y = prepare_features(featured)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert not any(col.startswith("customerID_") for col in X.columns)
    assert set(y.unique()) <= {0, 1}
    assert y.tolist() == [0, 1, 0, 0]


def test_fill_missing_numeric_uses_train_median_only():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, None]})
    test = pd.DataFrame({"x": [None, 100.0]})

    filled_train, filled_test = fill_missing_numeric(train, ["x"], test)

    # median of [1, 2, 3] is 2.0 - the test frame's real value (100.0) must
    # NOT influence what gets imputed anywhere (no leakage)
    assert filled_train["x"].iloc[-1] == 2.0
    assert filled_test["x"].iloc[0] == 2.0
    assert filled_test["x"].iloc[1] == 100.0
