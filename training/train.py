"""Train the churn model and export everything the API needs.

Run from the repository root: python training/train.py
Writes model/churn_model.onnx, model/schema.json, model/parity_cases.json and docs/EXPERIMENTS.md.
"""
import hashlib
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clean import clean_customers

ROOT = Path(__file__).resolve().parent.parent
# Pinned to one commit so the data cannot change under the results.
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv"
)
RANDOM_STATE = 42
MEDIUM_RISK = 0.3
HIGH_RISK = 0.6
MAX_EXPORT_DIFFERENCE = 1e-3

NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
# Limits of what could be a real customer, wider than what the training data contains.
NUMERIC_LIMITS = {
    "SeniorCitizen": {"min": 0, "max": 1},
    "tenure": {"min": 0, "max": 120},
    "MonthlyCharges": {"min": 0, "max": 1000},
    "TotalCharges": {"min": 0, "max": 200000},
}


def load_data() -> pd.DataFrame:
    path = ROOT / "training" / "data" / "telco.csv"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        urlretrieve(DATA_URL, path)
    return pd.read_csv(path, dtype={"TotalCharges": str})


def band_table(probabilities: np.ndarray, churned: pd.Series) -> list[dict]:
    bands = pd.cut(probabilities, [-0.01, MEDIUM_RISK, HIGH_RISK, 1.01], labels=["low", "medium", "high"], right=False)
    rows = []
    for band in ["high", "medium", "low"]:
        mask = np.asarray(bands == band)
        rows.append({
            "band": band,
            "customers_share": float(mask.mean()),
            "churn_rate": float(churned[mask].mean()),
            "share_of_all_churners": float(churned[mask].sum() / churned.sum()),
        })
    return rows


def main() -> None:
    raw = load_data()
    data = clean_customers(raw)
    blanks = int(pd.to_numeric(raw["TotalCharges"], errors="coerce").isna().sum())
    print(f"Rows: {len(data)} | blank TotalCharges fixed: {blanks} | churn rate: {data['Churn'].mean():.3f}")

    features = [column for column in data.columns if column != "Churn"]
    categorical = [column for column in features if column not in NUMERIC]
    X_train, X_test, y_train, y_test = train_test_split(
        data[features], data["Churn"], test_size=0.2, random_state=RANDOM_STATE, stratify=data["Churn"]
    )

    def pipeline(model):
        return Pipeline([
            ("preprocess", ColumnTransformer([
                ("numeric", StandardScaler(), NUMERIC),
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ])),
            ("model", model),
        ])

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    initial_types = [
        (column, FloatTensorType([None, 1]) if column in NUMERIC else StringTensorType([None, 1]))
        for column in features
    ]

    def export(fitted) -> bytes:
        options = {type(fitted.named_steps["model"]): {"zipmap": False}}
        return convert_sklearn(fitted, initial_types=initial_types, options=options, target_opset=17).SerializeToString()

    def onnx_probabilities(onnx_bytes: bytes, X: pd.DataFrame) -> np.ndarray:
        session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
        feed = {
            column: X[[column]].to_numpy(dtype=np.float32 if column in NUMERIC else object)
            for column in features
        }
        return session.run(["probabilities"], feed)[0][:, 1]

    # The model is chosen on the training set only. The test set is scored once, below.
    # A model the API cannot reproduce is not a candidate, however well it scores.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = []
    out_of_fold = {}
    for name, model in candidates.items():
        probabilities = cross_val_predict(pipeline(model), X_train, y_train, cv=folds, method="predict_proba")[:, 1]
        out_of_fold[name] = probabilities
        # Checked on customers each model did not train on: trees can match on their own
        # training rows and still disagree on new values that fall close to a split.
        worst_export_difference = 0.0
        for fit_rows, check_rows in folds.split(X_train, y_train):
            fitted = pipeline(model).fit(X_train.iloc[fit_rows], y_train.iloc[fit_rows])
            check = X_train.iloc[check_rows]
            difference = np.abs(onnx_probabilities(export(fitted), check) - fitted.predict_proba(check)[:, 1])
            worst_export_difference = max(worst_export_difference, float(difference.max()))
        comparison.append({
            "model": name,
            "roc_auc": roc_auc_score(y_train, probabilities),
            "pr_auc": average_precision_score(y_train, probabilities),
            "f1": f1_score(y_train, probabilities >= 0.5),
            "export_difference": worst_export_difference,
        })
        print(comparison[-1])
    comparison.sort(key=lambda row: row["roc_auc"], reverse=True)
    eligible = [row for row in comparison if row["export_difference"] <= MAX_EXPORT_DIFFERENCE]
    if not eligible:
        raise RuntimeError("No candidate survives export to ONNX")
    best_name = eligible[0]["model"]

    best = pipeline(candidates[best_name]).fit(X_train, y_train)
    test_probabilities = best.predict_proba(X_test)[:, 1]
    test_predictions = test_probabilities >= 0.5
    test_metrics = {
        "roc_auc": roc_auc_score(y_test, test_probabilities),
        "pr_auc": average_precision_score(y_test, test_probabilities),
        "accuracy": accuracy_score(y_test, test_predictions),
        "precision": precision_score(y_test, test_predictions),
        "recall": recall_score(y_test, test_predictions),
        "f1": f1_score(y_test, test_predictions),
    }
    print("Chosen:", best_name, test_metrics)

    bands = band_table(test_probabilities, y_test.reset_index(drop=True))
    top_fifth = np.argsort(-test_probabilities)[: len(test_probabilities) // 5]
    top_fifth_capture = float(y_test.iloc[top_fifth].sum() / y_test.sum())
    importance = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE)
    drivers = sorted(zip(features, importance.importances_mean), key=lambda pair: pair[1], reverse=True)[:6]

    model_dir = ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    onnx_bytes = export(best)
    (model_dir / "churn_model.onnx").write_bytes(onnx_bytes)

    # The exported model must give the same answers as the trained one, or the API serves a different model.
    worst_difference = float(np.abs(onnx_probabilities(onnx_bytes, X_test) - test_probabilities).max())
    print("Largest difference between scikit-learn and ONNX on the test set:", worst_difference)
    if worst_difference > MAX_EXPORT_DIFFERENCE:
        raise RuntimeError("The ONNX model disagrees with the trained model")

    version = f"{hashlib.sha256(onnx_bytes).hexdigest()[:8]}"
    schema = {
        "modelVersion": version,
        "modelName": best_name,
        "numeric": {column: NUMERIC_LIMITS[column] for column in NUMERIC},
        "categorical": {column: sorted(data[column].unique().tolist()) for column in categorical},
        "riskBands": {"medium": MEDIUM_RISK, "high": HIGH_RISK},
        "monitoring": {
            "baselineMeanProbability": round(float(out_of_fold[best_name].mean()), 4),
            "driftTolerance": 0.10,
            "minimumPredictions": 30,
        },
    }
    (model_dir / "schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8", newline="\n")

    cases = X_test.head(25).to_dict(orient="records")
    for case, probability in zip(cases, test_probabilities[:25]):
        case["expectedProbability"] = round(float(probability), 6)
    (model_dir / "parity_cases.json").write_text(json.dumps(cases, indent=1) + "\n", encoding="utf-8", newline="\n")

    lines = [
        "# Experiments",
        "",
        "Generated by `training/train.py`. Do not edit by hand.",
        "",
        f"**Data:** IBM Telco customer churn sample, {len(data):,} customers, {data['Churn'].mean():.1%} churned. "
        "This is a fictional sample dataset, so the numbers show the method and not a real business.",
        "",
        "**Cleaning:** `TotalCharges` arrives as text. "
        f"{blanks} blanks all belong to customers in their first month, so they become 0. "
        "A blank for any other customer stops training. `customerID` and `gender` are not used as features.",
        "",
        "**Method:** 80/20 stratified split, done first. Scaling and encoding live inside the pipeline, "
        "so they are fit on training data only. The model is chosen with 5-fold cross-validation on the training set, "
        "and the test set is scored once.",
        "",
        "## Model choice (5-fold cross-validation, training set)",
        "",
        "| Model | ROC AUC | PR AUC | F1 at 0.5 | Largest difference after ONNX export |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {row['model']} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} | {row['f1']:.3f} | {row['export_difference']:.1e} |"
        for row in comparison
    ]
    lines += [
        "",
        f"Chosen: **{best_name}**, the highest ROC AUC among models whose ONNX export stays within "
        f"{MAX_EXPORT_DIFFERENCE} of the trained model, measured in cross-validation on customers it did not train on. "
        "A model that changes when exported is not the model that was evaluated, so it is not eligible, "
        "even if it scores slightly higher.",
        "",
        f"## Final result (test set, {len(X_test):,} customers, scored once)",
        "",
        "| ROC AUC | PR AUC | Accuracy | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|---:|---:|",
        "| " + " | ".join(f"{test_metrics[key]:.3f}" for key in ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]) + " |",
        "",
        "## What a retention team would do with it",
        "",
        f"Calling the 20% of customers with the highest scores reaches **{top_fifth_capture:.0%} of the customers who actually churned**.",
        "",
        "| Risk band | Share of customers | Churn rate | Share of all churners |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {row['band']} | {row['customers_share']:.0%} | {row['churn_rate']:.0%} | {row['share_of_all_churners']:.0%} |"
        for row in bands
    ]
    lines += [
        "",
        "## Features that matter most",
        "",
        "Drop in test ROC AUC when one feature is shuffled (permutation importance).",
        "",
        "| Feature | Drop in ROC AUC |",
        "|---|---:|",
    ]
    lines += [f"| {feature} | {value:.3f} |" for feature, value in drivers]
    lines += [
        "",
        "## Export check",
        "",
        f"The largest difference between scikit-learn and the exported ONNX model on the whole test set is {worst_difference:.1e}. "
        "Training fails if it passes 0.001. The .NET tests repeat the check on 25 customers.",
        "",
    ]
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "EXPERIMENTS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
