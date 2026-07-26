"""Tune a classifier and log parameters, metrics, model, and plots manually."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", str(BASE_DIR / ".cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


TARGET = "diagnosis"
RANDOM_STATE = 42
DAGSHUB_OWNER = "petershaan12"
DAGSHUB_REPOSITORY = "SMSML-Peter-Shaan"


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    return frame.drop(columns=TARGET), frame[TARGET]


def save_evaluation_artifacts(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_precision": precision_score(y_test, predictions),
        "test_recall": recall_score(y_test, predictions),
        "test_f1": f1_score(y_test, predictions),
        "test_roc_auc": roc_auc_score(y_test, probabilities),
    }

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["malignant", "benign"], cmap="Blues"
    )
    plt.title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title("ROC Curve — Test Set")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    importance = pd.DataFrame(
        {"feature": X_test.columns, "coefficient": model.coef_[0]}
    ).assign(abs_coefficient=lambda frame: frame["coefficient"].abs())
    importance.sort_values("abs_coefficient", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(classification_report(y_test, predictions, output_dict=True), indent=2),
        encoding="utf-8",
    )
    joblib.dump(model, output_dir / "best_model.joblib")
    return metrics


def configure_tracking(tracking: str) -> None:
    if tracking == "local":
        mlflow.set_tracking_uri((BASE_DIR / "mlruns").resolve().as_uri())
        return

    token = os.getenv("DAGSHUB_USER_TOKEN")
    if not token:
        raise RuntimeError(
            "DAGSHUB_USER_TOKEN is not set. Add DAGSHUB_TOKEN to Colab Secrets "
            "and expose it as DAGSHUB_USER_TOKEN before running this script."
        )

    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_OWNER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    import dagshub

    dagshub.init(
        repo_owner=DAGSHUB_OWNER,
        repo_name=DAGSHUB_REPOSITORY,
        mlflow=True,
    )


def train(data_dir: Path, artifact_dir: Path, tracking: str) -> str:
    X_train, y_train = load_split(data_dir / "train.csv")
    X_test, y_test = load_split(data_dir / "test.csv")

    search = GridSearchCV(
        estimator=LogisticRegression(max_iter=3_000, random_state=RANDOM_STATE),
        param_grid={
            "C": [0.01, 0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        },
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    metrics = save_evaluation_artifacts(best_model, X_test, y_test, artifact_dir)

    configure_tracking(tracking)
    mlflow.set_experiment("breast-cancer-tuning")
    with mlflow.start_run(run_name="logistic-regression-grid-search") as run:
        mlflow.log_params(search.best_params_)
        mlflow.log_params(
            {
                "cv_folds": 5,
                "scoring": "roc_auc",
                "training_rows": len(X_train),
                "feature_count": X_train.shape[1],
                "random_state": RANDOM_STATE,
            }
        )
        mlflow.log_metric("best_cv_roc_auc", search.best_score_)
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(artifact_dir), artifact_path="evaluation")
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            signature=infer_signature(X_train, best_model.predict(X_train)),
            input_example=X_train.head(3),
        )
        return run.info.run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE_DIR / "breast_cancer_preprocessing",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=BASE_DIR / "artifacts",
    )
    parser.add_argument(
        "--tracking",
        choices=("local", "dagshub"),
        default="local",
        help="Use local MLflow storage or the configured DagsHub repository.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_id = train(args.data_dir, args.artifact_dir, args.tracking)
    print(f"MLflow tuning run completed: {run_id}")
