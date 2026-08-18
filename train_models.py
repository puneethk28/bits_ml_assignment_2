"""
Training script: Loads Adult Income CSV, trains all 5 models,
saves preprocessor + model .pkl files, and exports test_data.csv.
"""

import os
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

os.makedirs("model", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="adult_income.csv", help="Path to the input CSV dataset")
CSV_PATH = parser.parse_args().data

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} not found in project root.")

print(f"Loading dataset from {CSV_PATH} …")
df = pd.read_csv(CSV_PATH)

X: pd.DataFrame = df.drop(columns=["income"])
y = df["income"].astype(int)

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols   = X.select_dtypes(include="number").columns.tolist()

print(f"Features  : {X.shape[1]}  ({len(numerical_cols)} numeric, {len(categorical_cols)} categorical)")
print(f"Instances : {X.shape[0]}")

# ── 3. Train / test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── 4. Preprocessor ───────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)
preprocessor.fit(X_train)

X_train_proc = preprocessor.transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

joblib.dump(preprocessor, "model/preprocessor.pkl", compress=3)
print("Preprocessor saved → model/preprocessor.pkl")

# ── 5. Save column metadata so the app can validate uploads ─────────────────
meta = {
    "numerical_cols":   numerical_cols,
    "categorical_cols": categorical_cols,
    "all_feature_cols": numerical_cols + categorical_cols,
    "target_col":       "income",
}
joblib.dump(meta, "model/meta.pkl", compress=3)

# ── 6. Train models ───────────────────────────────────────────────────────────
models_to_train = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes":         GaussianNB(),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}

FILE_NAMES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree":       "decision_tree",
    "KNN":                 "knn",
    "Naive Bayes":         "naive_bayes",
    "Random Forest":       "random_forest",
}

results = {}

for name, model in models_to_train.items():
    print(f"Training {name} …")
    model.fit(X_train_proc, y_train)

    joblib.dump(model, f"model/{FILE_NAMES[name]}.pkl", compress=3)

    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)[:, 1]

    results[name] = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "AUC":       round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "F1":        round(f1_score(y_test, y_pred), 4),
        "MCC":       round(matthews_corrcoef(y_test, y_pred), 4),
    }
    print(f"  Accuracy={results[name]['Accuracy']}  AUC={results[name]['AUC']}")

# ── 7. Save baseline results and test data ────────────────────────────────────
results_df = pd.DataFrame(results).T
results_df.to_csv("model/baseline_results.csv")
print("\n── Baseline results on held-out test set ──")
print(results_df.to_string())

test_df = X_test.copy()
test_df["income"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print(f"\nTest data saved → test_data.csv  ({len(test_df)} rows)")
print("\nAll done!  Run:  streamlit run app.py")
