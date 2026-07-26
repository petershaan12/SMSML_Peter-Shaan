"""Train the Basic model with MLflow autologging."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", str(BASE_DIR / ".cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


TARGET = "diagnosis"


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path)
    return frame.drop(columns=TARGET), frame[TARGET]


def train(data_dir: Path) -> str:
    X_train, y_train = load_split(data_dir / "train.csv")
    X_test, y_test = load_split(data_dir / "test.csv")

    mlflow.set_tracking_uri((BASE_DIR / "mlruns").resolve().as_uri())
    mlflow.set_experiment("breast-cancer-basic")
    mlflow.sklearn.autolog(log_models=True)
    with mlflow.start_run(run_name="logistic-regression-autolog") as run:
        model = LogisticRegression(max_iter=2_000, random_state=42, solver="liblinear")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        mlflow.log_metrics(
            {
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_precision": precision_score(y_test, predictions),
                "test_recall": recall_score(y_test, predictions),
                "test_f1": f1_score(y_test, predictions),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
            }
        )
        return run.info.run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE_DIR / "breast_cancer_preprocessing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_id = train(args.data_dir)
    print(f"MLflow run completed: {run_id}")
