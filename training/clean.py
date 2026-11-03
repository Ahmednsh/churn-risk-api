"""Cleaning rules for the IBM Telco customer churn dataset."""
import pandas as pd

NOT_FEATURES = ["customerID", "gender"]


def clean_customers(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.drop_duplicates().copy()

    # TotalCharges arrives as text. It is blank only for customers in their first
    # month, who have not been charged yet, so blank means 0 for them. A blank for
    # anyone else would be a real data problem, so training stops instead of guessing.
    charges = pd.to_numeric(data["TotalCharges"], errors="coerce")
    if (data.loc[charges.isna(), "tenure"] > 0).any():
        raise ValueError("TotalCharges is blank for a customer whose tenure is above 0")
    data["TotalCharges"] = charges.fillna(0.0)

    if not data["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Churn must be Yes or No")
    data["Churn"] = (data["Churn"] == "Yes").astype(int)

    return data.drop(columns=NOT_FEATURES).reset_index(drop=True)
