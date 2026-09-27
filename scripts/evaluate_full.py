"""Evaluate the saved full-data SVD model on its own training ratings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.model import SVDModel


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Exam" / "data" / "ml-latest-small-filtered"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = actual - predicted
    return {
        "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "mae": round(float(np.mean(np.abs(errors))), 4),
    }


def main() -> None:
    ratings_path = DATA_DIR / "ratings.csv"
    metadata = json.loads((ARTIFACT_DIR / "svd_metadata.json").read_text(encoding="utf-8"))
    if hashlib.sha256(ratings_path.read_bytes()).hexdigest() != metadata["ratings_sha256"]:
        raise RuntimeError("Ratings changed since training; retrain first")
    ratings = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"])
    model = SVDModel.load(str(ARTIFACT_DIR / "svd_model.npz"))
    if len(ratings) != metadata["ratings_count"]:
        raise RuntimeError("Ratings count differs from saved training metadata")

    user_index = pd.Series(np.arange(len(model.user_ids)), index=model.user_ids)
    movie_index = pd.Series(np.arange(len(model.movie_ids)), index=model.movie_ids)
    users = ratings.userId.map(user_index)
    movies = ratings.movieId.map(movie_index)
    if users.isna().any() or movies.isna().any():
        raise RuntimeError("A training user or movie is missing from the saved model")
    u = users.to_numpy(dtype=np.int64)
    i = movies.to_numpy(dtype=np.int64)
    actual = ratings.rating.to_numpy(dtype=float)
    predicted = np.clip(
        model.global_mean
        + model.user_bias[u]
        + model.movie_bias[i]
        + np.einsum("ij,ij->i", model.user_factors[u], model.movie_factors[i]),
        0.5,
        5.0,
    )
    user_mean = ratings.userId.map(ratings.groupby("userId").rating.mean()).to_numpy(dtype=float)
    movie_count = ratings.groupby("movieId").size()
    sparse_mask = ratings.movieId.map(movie_count).to_numpy() < 5

    report = {
        "evaluation_type": "in_sample_on_full_training_data",
        "warning": "These ratings were used to fit the saved model. This measures training fit, not generalization or top-K recommendation quality.",
        "ratings_count": len(ratings),
        "users_count": int(ratings.userId.nunique()),
        "movies_count": int(ratings.movieId.nunique()),
        "svd": error_metrics(actual, predicted),
        "user_mean_baseline": error_metrics(actual, user_mean),
        "movies_with_fewer_than_5_ratings": {
            "rating_rows": int(sparse_mask.sum()),
            "svd": error_metrics(actual[sparse_mask], predicted[sparse_mask]),
        },
        "movies_with_at_least_5_ratings": {
            "rating_rows": int((~sparse_mask).sum()),
            "svd": error_metrics(actual[~sparse_mask], predicted[~sparse_mask]),
        },
    }
    output = ARTIFACT_DIR / "full_train_evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
