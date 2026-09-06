"""
train_model.py
--------------
Loads the HR Attrition dataset, preprocesses it, compares multiple
classification models, selects the best by ROC-AUC, and saves the
complete pipeline to models/model.pkl.
"""

import os
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
)

warnings.filterwarnings("ignore")

# ── 1. Load data ─────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "dataset attrition.csv")
df = pd.read_csv(DATA_PATH)

# Drop fully-empty column and date/leakage columns
DROP_COLS = [
    "Unnamed: 32",
    "Date_of_Hire",
    "Date_of_termination",   # all NaN → no info
    "Status_of_leaving",     # post-event leakage
]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Target ─────────────────────────────────────────────────────────────────
TARGET = "Attrition"
df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Target distribution:\n{y.value_counts().to_string()}\n")

# ── 3. Feature identification ─────────────────────────────────────────────────
CATEGORICAL_FEATURES = X.select_dtypes(include=["object"]).columns.tolist()
NUMERICAL_FEATURES   = X.select_dtypes(include=["number"]).columns.tolist()

print(f"Numerical features  ({len(NUMERICAL_FEATURES)}): {NUMERICAL_FEATURES}")
print(f"Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}\n")

# ── 4. Preprocessing pipelines ────────────────────────────────────────────────
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, NUMERICAL_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

# ── 5. Model definitions ──────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=200, random_state=42
    ),
}

# ── 6. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── 7. Cross-validated comparison ────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 60)
print("5-Fold Cross-Validation ROC-AUC Scores")
print("=" * 60)

results = {}
for name, clf in MODELS.items():
    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    results[name] = scores.mean()
    print(f"  {name:<25}  AUC = {scores.mean():.4f}  (±{scores.std():.4f})")

best_name = max(results, key=results.get)
print(f"\nBest model: {best_name}  (AUC = {results[best_name]:.4f})\n")

# ── 8. Final training on full train set ───────────────────────────────────────
best_clf = MODELS[best_name]
best_pipeline = Pipeline(steps=[("pre", preprocessor), ("clf", best_clf)])
best_pipeline.fit(X_train, y_train)

# ── 9. Evaluation on held-out test set ───────────────────────────────────────
y_pred      = best_pipeline.predict(X_test)
y_pred_prob = best_pipeline.predict_proba(X_test)[:, 1]

print("=" * 60)
print(f"Test-Set Evaluation  —  {best_name}")
print("=" * 60)
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, y_pred_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Attrition", "Attrition"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── 10. Feature importance (if available) ────────────────────────────────────
feature_names = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
clf_step = best_pipeline.named_steps["clf"]
if hasattr(clf_step, "feature_importances_"):
    importances = clf_step.feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\nTop-10 Feature Importances:")
    print(fi.head(10).to_string())

# ── 11. Save pipeline + metadata ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
MODEL_PATH = os.path.join("models", "model.pkl")

metadata = {
    "pipeline":            best_pipeline,
    "best_model_name":     best_name,
    "numerical_features":  NUMERICAL_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "feature_names":       feature_names,
    "target":              TARGET,
    "categorical_options": {
        col: sorted(df[col].dropna().unique().tolist())
        for col in CATEGORICAL_FEATURES
    },
    "numerical_stats": {
        col: {
            "min":    float(df[col].min()),
            "max":    float(df[col].max()),
            "mean":   float(df[col].mean()),
            "median": float(df[col].median()),
        }
        for col in NUMERICAL_FEATURES
    },
    "test_roc_auc":  round(roc_auc_score(y_test, y_pred_prob), 4),
    "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
    "all_results":   results,
}

joblib.dump(metadata, MODEL_PATH)
print(f"\nModel + metadata saved: {MODEL_PATH}")
