import pandas as pd
import pytest

from clean import clean_customers


def raw_rows(**overrides):
    row = {
        "customerID": "0001-A",
        "gender": "Female",
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 50.0,
        "TotalCharges": "600.0",
        "Contract": "One year",
        "Churn": "No",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_blank_total_charges_for_a_brand_new_customer_become_zero():
    cleaned = clean_customers(raw_rows(tenure=0, TotalCharges=" "))
    assert cleaned.loc[0, "TotalCharges"] == 0.0


def test_total_charges_become_numbers():
    cleaned = clean_customers(raw_rows(TotalCharges="1889.5"))
    assert cleaned.loc[0, "TotalCharges"] == 1889.5


def test_blank_total_charges_for_an_existing_customer_are_rejected():
    with pytest.raises(ValueError, match="TotalCharges"):
        clean_customers(raw_rows(tenure=5, TotalCharges=" "))


def test_churn_becomes_one_or_zero():
    assert clean_customers(raw_rows(Churn="Yes")).loc[0, "Churn"] == 1
    assert clean_customers(raw_rows(Churn="No")).loc[0, "Churn"] == 0


def test_unknown_churn_value_is_rejected():
    with pytest.raises(ValueError, match="Churn"):
        clean_customers(raw_rows(Churn="Maybe"))


def test_identifier_and_gender_are_not_kept_as_features():
    cleaned = clean_customers(raw_rows())
    assert "customerID" not in cleaned.columns
    assert "gender" not in cleaned.columns


def test_duplicate_customers_are_dropped():
    two_rows = pd.concat([raw_rows(), raw_rows()], ignore_index=True)
    assert len(clean_customers(two_rows)) == 1
