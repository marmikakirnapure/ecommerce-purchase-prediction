from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    ConfusionMatrixDisplay, f1_score, precision_score, recall_score,
    roc_auc_score, RocCurveDisplay
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

def main():
    # ============================================================
    # MARKS 1: DATA CLEANING & SPLITTING (3 MARKS)
    # ============================================================
    # 1. Load and inspect data
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put the CSV in the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    original_rows, original_cols = df.shape
    missing_before = int(df.isna().sum().sum())
    duplicates_before = int(df.duplicated().sum())

    # 2. Data cleaning
    # Remove exact duplicate rows. The supplied file has no missing values.
    df = df.drop_duplicates().copy()
    duplicates_removed = original_rows - len(df)

    # Convert the boolean target robustly to 0/1.
    target_map = {
        True: 1, False: 0,
        "True": 1, "False": 0,
        "true": 1, "false": 0,
        1: 1, 0: 0
    }
    df["Revenue"] = df["Revenue"].map(target_map)
    if df["Revenue"].isna().any():
        raise ValueError("Unexpected values found in Revenue target.")
    df["Revenue"] = df["Revenue"].astype(int)

    # PageValues can be closely associated with completed transactions.
    # Exclude it from predictors to reduce outcome-related leakage.
    excluded_features = ["Revenue", "PageValues"]
    X = df.drop(columns=excluded_features)
    y = df["Revenue"]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "bool", "category"]
    ).columns.tolist()

    # 3. EDA figures and cleaning summary
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    plt.bar(["No Purchase", "Purchase"], counts.values)
    plt.title("Revenue Class Distribution")
    plt.ylabel("Number of sessions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=250)
    plt.close()

    numeric_for_corr = df.select_dtypes(include=["number"])
    if not numeric_for_corr.empty:
        plt.figure(figsize=(11, 8))
        sns.heatmap(numeric_for_corr.corr(), cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap (Numeric Columns)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=250)
        plt.close()

    summary = {
        "original_rows": int(original_rows),
        "original_columns": int(original_cols),
        "missing_values_before_cleaning": missing_before,
        "duplicate_rows_detected": duplicates_before,
        "duplicate_rows_removed": int(duplicates_removed),
        "rows_after_cleaning": int(len(df)),
        "missing_values_after_cleaning": int(df.isna().sum().sum()),
        "purchase_sessions": int((y == 1).sum()),
        "non_purchase_sessions": int((y == 0).sum()),
        "purchase_percentage": round(float(y.mean() * 100), 3),
        "target": "Revenue",
        "excluded_predictors": excluded_features,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "scikit_learn_version": sklearn.__version__,
    }
    with open(OUTPUT_DIR / "cleaning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 4. Split before fitting any learned preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 5. Preprocessing stays inside each pipeline to prevent leakage.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    # ============================================================
    # MARKS 2: MODEL COMPARISON & PERFORMANCE ANALYTICS (4 MARKS)
    # ============================================================
    # Exactly four models from the professor's approved example list.
    # Hyperparameters are selected using 3-fold CV on training data only.
    model_grids = {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=2000, class_weight="balanced",
                random_state=RANDOM_STATE
            ),
            {"classifier__C": [0.1, 1.0, 10.0]}
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "classifier__n_neighbors": [5, 9, 15],
                "classifier__weights": ["uniform", "distance"]
            }
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            {
                "classifier__max_depth": [5, 10, None],
                "classifier__min_samples_leaf": [1, 5]
            }
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1
            ),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 15],
                "classifier__min_samples_leaf": [1, 3]
            }
        ),
    }

    cv = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=RANDOM_STATE
    )
    fitted_models = {}
    rows = []

    print(f"Dataset after duplicate removal: {df.shape}")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("Training and tuning exactly four models. This may take a few minutes.")

    for model_name, (classifier, param_grid) in model_grids.items():
        print(f"\nTuning {model_name}...")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        predictions = best_pipeline.predict(X_test)
        probabilities = best_pipeline.predict_proba(X_test)[:, 1]

        row = {
            "Model": model_name,
            "CV_F1_Mean": float(search.best_score_),
            "Best_Params": str(search.best_params_),
            "Accuracy": float(accuracy_score(y_test, predictions)),
            "Precision_Purchase": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "Recall_Purchase": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "F1_Purchase": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "ROC_AUC": float(roc_auc_score(y_test, probabilities)),
            "PR_AUC_Average_Precision": float(
                average_precision_score(y_test, probabilities)
            ),
        }
        rows.append(row)
        fitted_models[model_name] = best_pipeline
        print(f"Best parameters: {search.best_params_}")
        print(f"CV F1: {search.best_score_:.4f} | Test F1: {row['F1_Purchase']:.4f}")

    results = pd.DataFrame(rows).sort_values(
        "CV_F1_Mean", ascending=False
    ).reset_index(drop=True)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # Select using training CV, not the test set.
    best_name = results.loc[0, "Model"]
    best_model = fitted_models[best_name]
    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "features": list(X.columns),
        "excluded_features": excluded_features,
        "sklearn_version": sklearn.__version__,
    }, MODEL_DIR / "best_model.joblib")

    # Save a detailed report for the selected model.
    best_predictions = best_model.predict(X_test)
    best_probabilities = best_model.predict_proba(X_test)[:, 1]
    report = classification_report(
        y_test, best_predictions,
        labels=[0, 1],
        target_names=["No Purchase", "Purchase"],
        output_dict=True,
        zero_division=0
    )
    with open(OUTPUT_DIR / "best_model_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    ConfusionMatrixDisplay.from_predictions(
        y_test, best_predictions,
        labels=[0, 1],
        display_labels=["No Purchase", "Purchase"],
        cmap="Blues"
    )
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=250)
    plt.close()

    RocCurveDisplay.from_predictions(
        y_test, best_probabilities, name=best_name
    )
    plt.plot([0, 1], [0, 1], "--", label="Random baseline")
    plt.title(f"ROC Curve — {best_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=250)
    plt.close()

    plt.figure(figsize=(9, 5))
    plot_data = results.sort_values("F1_Purchase", ascending=False)
    plt.bar(plot_data["Model"], plot_data["F1_Purchase"])
    plt.title("Four-Model Comparison — Test Purchase F1-Score")
    plt.ylabel("F1-score for Purchase class")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=250)
    plt.close()

    with open(OUTPUT_DIR / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected_model_by_training_cv_f1": best_name,
            "selection_metric": "Best mean 3-fold stratified CV F1 on training data",
            "test_metrics_are_held_out": True,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "excluded_features": excluded_features,
            "scikit_learn_version": sklearn.__version__,
        }, f, indent=2)

    print("\n===== FINAL TEST-SET COMPARISON =====")
    print(results[[
        "Model", "CV_F1_Mean", "Accuracy", "Precision_Purchase",
        "Recall_Purchase", "F1_Purchase", "ROC_AUC",
        "PR_AUC_Average_Precision"
    ]].round(4).to_string(index=False))
    print(f"\nSelected model by training CV F1: {best_name}")
    print(f"Saved trained model to: {MODEL_DIR / 'best_model.joblib'}")
    print(f"Saved metrics and figures to: {OUTPUT_DIR}")

    # ============================================================
    # MARKS 3: UI DEPLOYMENT & CROSS-PLATFORM ACCESSIBILITY (4 MARKS)
    # The Streamlit interface is in app.py. Run: streamlit run app.py
    # Test the local app on desktop and mobile, then deploy using the guide.
    # ============================================================

if __name__ == "__main__":
    main()
