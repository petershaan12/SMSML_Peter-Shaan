"""Send raw dataset rows to the prediction API to generate monitoring traffic."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.datasets import load_breast_cancer


BASE_DIR = Path(__file__).resolve().parent


def load_input_frame(dataset: Path | None) -> pd.DataFrame:
    if dataset is not None:
        return pd.read_csv(dataset).drop(columns="diagnosis", errors="ignore")
    return load_breast_cancer(as_frame=True).frame.drop(columns="target")


def main(url: str, dataset: Path | None, count: int, delay: float) -> None:
    frame = load_input_frame(dataset)
    feature_names = requests.get(f"{url}/features", timeout=10).json()["features"]
    frame.columns = [
        column.lower().replace(" ", "_").replace("-", "_") for column in frame.columns
    ]
    for index, row in frame[feature_names].head(count).iterrows():
        response = requests.post(
            f"{url}/predict",
            json={"features": row.tolist()},
            timeout=10,
        )
        response.raise_for_status()
        print(index, response.json())
        time.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()
    main(args.url.rstrip("/"), args.dataset, args.count, args.delay)
