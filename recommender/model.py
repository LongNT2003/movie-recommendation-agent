"""Biased matrix factorization for explicit MovieLens ratings (Funk SVD)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SVDModel:
    user_ids: np.ndarray
    movie_ids: np.ndarray
    user_factors: np.ndarray
    movie_factors: np.ndarray
    user_bias: np.ndarray
    movie_bias: np.ndarray
    global_mean: float

    def predict(self, user_id: int, movie_id: int) -> float:
        users = {int(value): index for index, value in enumerate(self.user_ids)}
        movies = {int(value): index for index, value in enumerate(self.movie_ids)}
        if user_id not in users:
            raise ValueError(f"Unknown user_id: {user_id}")
        if movie_id not in movies:
            raise ValueError(f"Unknown movie_id in trained model: {movie_id}")
        u, i = users[user_id], movies[movie_id]
        score = self.global_mean + self.user_bias[u] + self.movie_bias[i]
        score += float(self.user_factors[u] @ self.movie_factors[i])
        return float(np.clip(score, 0.5, 5.0))

    def scores_for_user(self, user_id: int) -> np.ndarray:
        matches = np.flatnonzero(self.user_ids == user_id)
        if not len(matches):
            raise ValueError(f"Unknown user_id: {user_id}")
        u = int(matches[0])
        scores = (
            self.global_mean
            + self.user_bias[u]
            + self.movie_bias
            + self.movie_factors @ self.user_factors[u]
        )
        return np.clip(scores, 0.5, 5.0)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            user_ids=self.user_ids,
            movie_ids=self.movie_ids,
            user_factors=self.user_factors,
            movie_factors=self.movie_factors,
            user_bias=self.user_bias,
            movie_bias=self.movie_bias,
            global_mean=np.array(self.global_mean),
        )

    @classmethod
    def load(cls, path: str) -> SVDModel:
        with np.load(path, allow_pickle=False) as data:
            return cls(**{key: data[key] for key in data.files})


def fit_svd(
    ratings: pd.DataFrame,
    *,
    factors: int = 32,
    epochs: int = 20,
    learning_rate: float = 0.01,
    regularization: float = 0.05,
    seed: int = 42,
) -> SVDModel:
    """Fit bias + user/item latent factors with stochastic gradient descent."""
    if ratings.empty:
        raise ValueError("Ratings must not be empty")
    if factors < 1 or epochs < 1 or learning_rate <= 0 or regularization < 0:
        raise ValueError("Invalid SVD hyperparameters")

    user_ids, user_index = np.unique(ratings["userId"].to_numpy(dtype=np.int64), return_inverse=True)
    movie_ids, movie_index = np.unique(ratings["movieId"].to_numpy(dtype=np.int64), return_inverse=True)
    values = ratings["rating"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    user_factors = rng.normal(0, 0.1, size=(len(user_ids), factors))
    movie_factors = rng.normal(0, 0.1, size=(len(movie_ids), factors))
    user_bias = np.zeros(len(user_ids), dtype=np.float64)
    movie_bias = np.zeros(len(movie_ids), dtype=np.float64)
    global_mean = float(values.mean())

    for _ in range(epochs):
        for row in rng.permutation(len(values)):
            u, i = user_index[row], movie_index[row]
            pu, qi = user_factors[u].copy(), movie_factors[i].copy()
            estimate = global_mean + user_bias[u] + movie_bias[i] + pu @ qi
            error = values[row] - estimate
            user_bias[u] += learning_rate * (error - regularization * user_bias[u])
            movie_bias[i] += learning_rate * (error - regularization * movie_bias[i])
            user_factors[u] += learning_rate * (error * qi - regularization * pu)
            movie_factors[i] += learning_rate * (error * pu - regularization * qi)

    return SVDModel(
        user_ids, movie_ids, user_factors, movie_factors,
        user_bias, movie_bias, global_mean,
    )
