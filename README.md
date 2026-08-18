# ML Assignment 2 – Classification Models with Streamlit

**Course:** Machine Learning · M.Tech (AIML/DSE) · BITS Pilani WILP  
**Submission Deadline:** 18-Aug-2026

---

## a. Problem Statement

Build an end-to-end binary (or multi-class) classification pipeline that:
1. Trains five supervised learning models on a labelled tabular dataset
2. Evaluates each model on six standard metrics
3. Exposes the results through an interactive Streamlit web application

**Example dataset used (Adult Income):** Predict whether an individual's annual income exceeds $50,000 based on demographic and employment attributes from the 1994 US Census data — `income` (0 = ≤50K, 1 = >50K).

The pipeline is dataset-agnostic: pass any CSV with a target column via `--data` and the system adapts automatically.

---

## b. Dataset Description

The pipeline requires a CSV dataset with:
- **Minimum 12 features** (mix of numeric and categorical supported)
- **Minimum 500 instances**
- A binary or multi-class **target column** (default assumed name: `income`; change in `train_models.py` if needed)

### Example Dataset — Adult Income (Census Income)

| Property | Value |
|---|---|
| **Name** | Adult Income (Census Income) |
| **Source** | UCI Machine Learning Repository |
| **Instances** | 45,222 (after removing rows with missing values) |
| **Features** | 14 (6 numeric, 8 categorical) |
| **Target** | `income` — binary (0: ≤$50K/yr, 1: >$50K/yr) |
| **Class distribution** | ~76% ≤50K · ~24% >50K | (Adult Income example)

| Feature | Type | Description |
|---|---|---|
| age | Numeric | Age of the individual |
| workclass | Categorical | Employment type (Private, Govt, Self-emp, etc.) |
| fnlwgt | Numeric | Census sampling weight |
| education | Categorical | Highest level of education attained |
| education_num | Numeric | Education level encoded as integer |
| marital_status | Categorical | Marital status |
| occupation | Categorical | Job/occupation category |
| relationship | Categorical | Role in family unit |
| race | Categorical | Race |
| sex | Categorical | Sex |
| capital_gain | Numeric | Investment capital gains |
| capital_loss | Numeric | Investment capital losses |
| hours_per_week | Numeric | Average working hours per week |
| native_country | Categorical | Country of origin |

**Preprocessing applied (same logic for any dataset):**
- Rows containing `?` (missing values) dropped — 3,620 rows removed in the Adult Income example
- Numerical features → `StandardScaler`
- Categorical features → `OneHotEncoder` (handle_unknown="ignore")
- Train/test split: 80% / 20% (stratified)

> Column names and dtypes are detected automatically from the CSV — no hardcoding required for new datasets.

---

## c. GitHub Repository Link

> **[https://github.com/YOUR_USERNAME/ml-assignment-2](https://github.com/YOUR_USERNAME/ml-assignment-2)**  
> *(Replace with your actual GitHub repository URL before submission)*

Repository structure:
```
ml-assignment-2/
├── app.py                  # Streamlit web application
├── train_models.py         # Model training script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── adult_income.csv        # Cleaned UCI Adult Income dataset (45,222 rows)
├── test_data.csv           # Held-out test set (9,045 rows)
└── model/
    ├── preprocessor.pkl      # Fitted StandardScaler + OneHotEncoder pipeline
    ├── meta.pkl              # Column metadata derived dynamically from the training CSV
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── baseline_results.csv
```

> **Note on `preprocessor.pkl` and `meta.pkl`:** Both files are required for the app to run.
> `preprocessor.pkl` transforms every uploaded CSV through the same fitted scaler and encoder used during training.
> `meta.pkl` stores the feature column names and target column name — derived dynamically from whatever CSV is passed to `train_models.py`, not hardcoded to the Adult Income dataset.
> The only fixed assumption is `target_col = "income"`; all column lists are inferred from the data at training time.
> If these files are missing, re-run `python train_models.py` to regenerate them.

---

## d. Models Used

### Comparison Table — Evaluation Metrics on Adult Income (20% held-out test set, 9,045 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8450 | 0.9020 | 0.7344 | 0.5870 | 0.6525 | 0.5601 |
| Decision Tree | 0.8473 | 0.8953 | 0.7776 | 0.5379 | 0.6359 | 0.5581 |
| kNN | 0.8302 | 0.8735 | 0.6781 | 0.5995 | 0.6364 | 0.5278 |
| Naive Bayes | 0.6193 | 0.8389 | 0.3877 | 0.9242 | 0.5462 | 0.3891 |
| Random Forest (Ensemble) | **0.8519** | **0.9047** | 0.7391 | **0.6218** | **0.6754** | **0.5840** |

---

### Model Observations (Adult Income dataset)

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Delivers strong, well-balanced performance (Accuracy 84.50%, AUC 0.9020). High AUC indicates excellent separation between classes even though the dataset is imbalanced. Precision (73.4%) is substantially higher than Recall (58.7%), meaning it correctly identifies high earners when it predicts them, but misses a notable fraction. A reliable baseline due to its simplicity and interpretability. |
| Decision Tree | Achieves the second-highest accuracy (84.73%) with the best Precision (77.76%) among all models, indicating very few false positives when it flags >50K earners. However, its Recall (53.79%) and AUC (0.8953) are weaker relative to Random Forest and Logistic Regression, suggesting the tree tends to be conservative and under-predicts the positive class. Prone to overfitting on deeper trees; max_depth=10 was used for regularisation. |
| kNN | Produces the most balanced Precision/Recall trade-off (67.81% / 59.95%) among the stronger models, giving it a competitive F1 (0.6364). However, its overall Accuracy (83.02%) and AUC (0.8735) are the lowest among the non-Naive-Bayes models. kNN is sensitive to the high-dimensional, mixed-type feature space produced by one-hot encoding, which explains the performance gap. Computationally expensive at inference time on large datasets. |
| Naive Bayes | Shows the starkest trade-off: extremely high Recall (92.42%) but very low Precision (38.77%), resulting in a large number of false positives. The Gaussian independence assumption is violated by correlated census features (e.g., education and education_num are near-duplicates), which degrades accuracy to 61.93%. Its AUC (0.8389) is still respectable, meaning the probability estimates carry discriminative signal even if the decision threshold produces many false alarms. Best used when missing a positive case is costlier than a false alarm. |
| Random Forest (Ensemble) | The best overall performer across Accuracy (85.19%), AUC (0.9047), Recall (62.18%), F1 (0.6754), and MCC (0.5840). Bagging over 200 trees reduces variance significantly compared to a single Decision Tree, improving generalisation. The ensemble naturally handles non-linear interactions (e.g., age × occupation × education) that linear models miss. Slightly lower Precision (73.91%) than Decision Tree, but the across-the-board metric gains make it the clear winner. |
| **Overall Winner for your dataset?** | **Random Forest** — it achieves the highest scores on 4 out of 6 metrics (Accuracy, AUC, Recall, F1, MCC) and provides the best balance between correctly identifying high-income earners and avoiding false positives on this imbalanced census dataset. |

---

## Streamlit App Features

- CSV upload (test data)
- Model selection dropdown (single model deep-dive)
- Compare all models simultaneously with a metric table and bar chart
- Confusion matrix visualisation for all models
- ROC curve with AUC for each model
- Detailed classification report (precision / recall per class)

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all 5 models (one-time, ~2 min)
python train_models.py                        # uses adult_income.csv by default
python train_models.py --data path/to/other.csv  # custom dataset

# 3. Launch the Streamlit app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) and upload `test_data.csv` from the sidebar.

## Live App

> **[https://APP.streamlit.app](https://APP.streamlit.app)**  

