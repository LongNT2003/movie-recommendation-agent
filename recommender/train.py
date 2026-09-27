"""Train and evaluate the SVD recommender, then fit the final artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.model import fit_svd


PROJECT_DIR = Path(__file__).resolve().parent.parent
RATINGS_PATH = PROJECT_DIR / "Exam" / "data" / "ml-latest-small-filtered" / "ratings.csv"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"


def split_per_user(ratings: pd.DataFrame, seed: int, fraction: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out at least one rating per user while retaining training history."""
    rng = np.random.default_rng(seed)
    validation_positions: list[int] = []
    for positions in ratings.groupby("userId", sort=True).indices.values():
        count = max(1, round(len(positions) * fraction))
        validation_positions.extend(rng.choice(positions, size=count, replace=False).tolist())
    mask = np.zeros(len(ratings), dtype=bool)
    mask[validation_positions] = True
    return ratings.loc[~mask].reset_index(drop=True), ratings.loc[mask].reset_index(drop=True)


def evaluate(model, train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    known_movies = set(model.movie_ids.tolist())
    covered = validation[validation.movieId.isin(known_movies)]
    predictions = np.array([model.predict(int(row.userId), int(row.movieId)) for row in covered.itertuples(index=False)])
    truth = covered.rating.to_numpy(dtype=float)
    user_means = train.groupby("userId").rating.mean()
    baseline = covered.userId.map(user_means).to_numpy(dtype=float)

    movie_to_index = {int(movie_id): index for index, movie_id in enumerate(model.movie_ids)}
    train_by_user = train.groupby("userId").movieId.apply(set).to_dict()
    movie_stats = train.groupby("movieId").rating.agg(["sum", "count"])
    min_ratings_for_top10 = 5
    eligible_movies = set(movie_stats.index[movie_stats["count"] >= min_ratings_for_top10].tolist())
    insufficient_rating_indices = [
        index for index, movie_id in enumerate(model.movie_ids) if movie_id not in eligible_movies
    ]
    popularity_scores = (
        (movie_stats["sum"] + 10 * model.global_mean) / (movie_stats["count"] + 10)
    ).reindex(model.movie_ids).to_numpy(dtype=float)
    positive_by_user = (
        validation[(validation.rating >= 4.0) & validation.movieId.isin(eligible_movies)]
        .groupby("userId").movieId.apply(set).to_dict()
    )
    hits = 0
    users_with_hit = 0
    popularity_hits = 0
    popularity_users_with_hit = 0
    eligible_users = 0
    relevant_total = 0
    for user_id, relevant in positive_by_user.items():
        eligible_users += 1
        relevant_total += len(relevant)
        scores = model.scores_for_user(int(user_id)).copy()
        seen_indices = [movie_to_index[movie_id] for movie_id in train_by_user[user_id]]
        scores[seen_indices] = -np.inf
        scores[insufficient_rating_indices] = -np.inf
        top_indices = np.argpartition(scores, -10)[-10:]
        recommended = set(model.movie_ids[top_indices].tolist())
        user_hits = len(recommended & relevant)
        hits += user_hits
        users_with_hit += int(user_hits > 0)
        baseline_scores = popularity_scores.copy()
        baseline_scores[seen_indices] = -np.inf
        baseline_scores[insufficient_rating_indices] = -np.inf
        baseline_top = np.argpartition(baseline_scores, -10)[-10:]
        baseline_hits = len(set(model.movie_ids[baseline_top].tolist()) & relevant)
        popularity_hits += baseline_hits
        popularity_users_with_hit += int(baseline_hits > 0)

    return {
        "validation_rows": int(len(validation)),
        "validation_covered_rows": int(len(covered)),
        "rmse": round(float(np.sqrt(np.mean((truth - predictions) ** 2))), 4),
        "mae": round(float(np.mean(np.abs(truth - predictions))), 4),
        "user_mean_baseline_rmse": round(float(np.sqrt(np.mean((truth - baseline) ** 2))), 4),
        "user_mean_baseline_mae": round(float(np.mean(np.abs(truth - baseline))), 4),
        "top10_relevant_rating_threshold": 4.0,
        "top10_candidate_min_ratings": min_ratings_for_top10,
        "top10_eligible_users": eligible_users,
        "top10_relevant_ratings": relevant_total,
        "hit_rate_at_10": round(users_with_hit / eligible_users, 4),
        "recall_at_10": round(hits / relevant_total, 4),
        "bayesian_movie_mean_baseline_hit_rate_at_10": round(popularity_users_with_hit / eligible_users, 4),
        "bayesian_movie_mean_baseline_recall_at_10": round(popularity_hits / relevant_total, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratings = pd.read_csv(RATINGS_PATH, usecols=["userId", "movieId", "rating"])
    if ratings.duplicated(["userId", "movieId"]).any():
        raise ValueError("Duplicate user/movie ratings found")
    train, validation = split_per_user(ratings, args.seed)
    hyperparameters = dict(
        factors=args.factors,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        seed=args.seed,
    )
    print(f"Validation fit: {len(train):,} train / {len(validation):,} held out")
    validation_model = fit_svd(train, **hyperparameters)
    metrics = evaluate(validation_model, train, validation)
    print(json.dumps(metrics, indent=2))
    print(f"Final fit: {len(ratings):,} ratings")
    final_model = fit_svd(ratings, **hyperparameters)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    validation_model.save(str(ARTIFACT_DIR / "svd_validation_model.npz"))
    final_model.save(str(ARTIFACT_DIR / "svd_model.npz"))
    metadata = {
        "algorithm": "biased matrix factorization (Funk SVD)",
        "rating_scale": [0.5, 5.0],
        "ratings_sha256": hashlib.sha256(RATINGS_PATH.read_bytes()).hexdigest(),
        "ratings_count": len(ratings),
        "users_count": int(ratings.userId.nunique()),
        "rated_movies_count": int(ratings.movieId.nunique()),
        "validation": metrics,
        "validation_method": "10% random held out per user, seed fixed; final model retrained on all ratings. Top-10 ranks all train-known, user-unseen movies; relevant means held-out rating >=4.",
        "hyperparameters": hyperparameters,
    }
    (ARTIFACT_DIR / "svd_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model and metadata to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
