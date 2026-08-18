"""
Streamlit app – Adult Income Classification
All 5 models are pre-trained; users upload test_data.csv for evaluation.
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Classification – Adult Income",
    page_icon="💰",
    layout="wide",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

DISPLAY_NAMES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree":       "decision_tree",
    "KNN":                 "knn",
    "Naive Bayes":         "naive_bayes",
    "Random Forest":       "random_forest",
}

METRIC_COLS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models …")
def load_artifacts():
    pre_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    meta_path = os.path.join(MODEL_DIR, "meta.pkl")
    if not os.path.exists(pre_path):
        return None, None, None
    preprocessor = joblib.load(pre_path)
    meta = joblib.load(meta_path)
    models = {}
    for name, fname in DISPLAY_NAMES.items():
        path = os.path.join(MODEL_DIR, f"{fname}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return preprocessor, models, meta


@st.cache_data(show_spinner=False)
def load_baseline():
    path = os.path.join(MODEL_DIR, "baseline_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────
def evaluate(model, X_proc, y_true):
    y_pred = model.predict(X_proc)
    y_prob = model.predict_proba(X_proc)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "AUC":       round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
    }, y_pred, y_prob


def metric_cards(metrics: dict):
    cols = st.columns(6)
    icons = {"Accuracy": "🎯", "AUC": "📈", "Precision": "🔍",
             "Recall": "🔁", "F1": "⚖️", "MCC": "🧮"}
    for col, key in zip(cols, METRIC_COLS):
        col.metric(f"{icons[key]} {key}", f"{metrics[key]:.4f}")


def highlight_best(df: pd.DataFrame):
    """Green background on the best value per column."""
    styled = df.style.format("{:.4f}")
    for col in df.columns:
        best = df[col].max()
        styled = styled.apply(
            lambda s, c=col, b=best: [
                "background-color: #d4edda; font-weight: bold" if v == b else ""
                for v in s
            ],
            subset=[col],
        )
    return styled


def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["≤50K", ">50K"],
        yticklabels=["≤50K", ">50K"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_roc(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_metric_bar(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(df))
    width = 0.13
    colors = sns.color_palette("tab10", len(METRIC_COLS))
    for i, metric in enumerate(METRIC_COLS):
        ax.bar(x + i * width, df[metric], width, label=metric, color=colors[i])
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(df.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison – All Metrics")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.title("💰 Adult Income – ML Classification Dashboard")
    st.markdown(
        "**Dataset:** UCI Adult Income (Census) &nbsp;|&nbsp; "
        "**Task:** Predict whether a person earns **>\\$50 K/yr** &nbsp;|&nbsp; "
        "**Models:** 5 classifiers compared"
    )
    st.divider()

    preprocessor, models, meta = load_artifacts()

    if preprocessor is None:
        st.error(
            "⚠️ Trained model files not found. "
            "Please run **`python train_models.py`** first to generate model artefacts."
        )
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Controls")
        uploaded = st.file_uploader(
            "📂 Upload Test Data (CSV)",
            type=["csv"],
            help="Upload test_data.csv generated by train_models.py",
        )
        st.divider()
        mode = st.radio(
            "View Mode",
            ["Compare All Models", "Single Model Deep-Dive"],
            index=0,
        )
        selected_model = None
        if mode == "Single Model Deep-Dive":
            selected_model = st.selectbox("Select Model", list(models.keys()))
        st.divider()
        st.caption("Adult Income Dataset · UCI / OpenML")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_results, tab_dataset, tab_about = st.tabs(
        ["📊 Results", "📋 Dataset and training Info", "ℹ️ About"]
    )

    # ── Tab: Dataset Training Info ───────────────────────────────────────────
    with tab_dataset:
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Source", "UCI / OpenML")
        col2.metric("Instances", "48,842")
        col3.metric("Features", "14")
        col4.metric("Target Classes", "2 (Binary)")
        st.markdown("""
        | Feature | Type | Description |
        |---------|------|-------------|
        | age | Numeric | Age of the individual |
        | workclass | Categorical | Employment type |
        | fnlwgt | Numeric | Census sampling weight |
        | education | Categorical | Highest education level |
        | education-num | Numeric | Education level as number |
        | marital-status | Categorical | Marital status |
        | occupation | Categorical | Occupation type |
        | relationship | Categorical | Relationship role |
        | race | Categorical | Race |
        | sex | Categorical | Sex |
        | capital-gain | Numeric | Capital gains |
        | capital-loss | Numeric | Capital losses |
        | hours-per-week | Numeric | Weekly work hours |
        | native-country | Categorical | Country of origin |
        | **income** | **Target** | **≤50K or >50K** |
        """)

        baseline = load_baseline()
        if baseline is not None:
            st.subheader("Baseline Results (training-time evaluation on 20% hold-out)")
            st.dataframe(highlight_best(baseline[METRIC_COLS]), use_container_width=True)
            st.pyplot(plot_metric_bar(baseline))

    # ── Tab: About ────────────────────────────────────────────────────────────
    with tab_about:
        st.subheader("Assignment – ML Classification Models")
        st.markdown("""
        **Models implemented:**
        1. Logistic Regression
        2. Decision Tree Classifier  
        3. K-Nearest Neighbor (KNN) Classifier
        4. Naive Bayes (Gaussian)
        5. Random Forest (Ensemble)

        **Evaluation Metrics:**  
        Accuracy · AUC · Precision · Recall · F1 Score · MCC

        **Preprocessing Pipeline:**
        - StandardScaler → numerical features
        - OneHotEncoder → categorical features
        - Missing values imputed with mode / median

        **How to use:**
        1. Upload `test_data.csv` using the sidebar
        2. Choose *Compare All Models* or drill into a single model
        3. Explore metrics, confusion matrices and ROC curves
        """)

    # ── Tab: Results ─────────────────────────────────────────────────────────
    with tab_results:
        if uploaded is None:
            st.info(
                "👈 Upload **test_data.csv** from the sidebar to evaluate models on your test set."
            )
            st.stop()

        # Load and validate CSV
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        target_col = meta["target_col"]
        if target_col not in df.columns:
            st.error(f"Uploaded CSV must contain a **'{target_col}'** column.")
            st.stop()

        X_raw = df.drop(columns=[target_col])
        y_true = df[target_col].astype(int)

        # Align columns
        expected = meta["all_feature_cols"]
        missing = set(expected) - set(X_raw.columns)
        if missing:
            st.error(f"Missing columns in uploaded file: {missing}")
            st.stop()

        # Convert categoricals to str (mirrors training)
        for col in meta["categorical_cols"]:
            if col in X_raw.columns:
                X_raw[col] = X_raw[col].astype(str)

        try:
            X_proc = preprocessor.transform(X_raw[expected])
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()

        st.success(f"✅ Loaded **{len(df):,}** test rows · Target distribution: "
                   f"{int((y_true == 0).sum())} ≤50K, {int((y_true == 1).sum())} >50K")

        # ── Compare All Models ────────────────────────────────────────────────
        if mode == "Compare All Models":
            st.subheader("📊 Comparison Table")
            all_results = {}
            all_preds = {}
            all_probs = {}
            for name, model in models.items():
                metrics, y_pred, y_prob = evaluate(model, X_proc, y_true)
                all_results[name] = metrics
                all_preds[name] = y_pred
                all_probs[name] = y_prob

            results_df = pd.DataFrame(all_results).T[METRIC_COLS]
            st.dataframe(highlight_best(results_df), use_container_width=True)

            # Bar chart
            st.pyplot(plot_metric_bar(results_df))

            # Confusion matrices
            st.subheader("Confusion Matrices")
            cols = st.columns(len(models))
            for col, (name, y_pred) in zip(cols, all_preds.items()):
                col.pyplot(plot_confusion(y_true, y_pred, title=name))

        # ── Single Model ──────────────────────────────────────────────────────
        else:
            model = models[selected_model]
            metrics, y_pred, y_prob = evaluate(model, X_proc, y_true)

            st.subheader(f"📌 {selected_model}")
            metric_cards(metrics)
            st.divider()

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Confusion Matrix")
                st.pyplot(plot_confusion(y_true, y_pred))
            with c2:
                st.subheader("ROC Curve")
                st.pyplot(plot_roc(y_true, y_prob, selected_model))

            st.subheader("Classification Report")
            report = classification_report(
                y_true, y_pred,
                target_names=["≤50K", ">50K"],
                output_dict=True,
            )
            report_df = pd.DataFrame(report).T.round(4)
            st.dataframe(report_df, use_container_width=True)


if __name__ == "__main__":
    main()
