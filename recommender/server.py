"""MCP stdio server for SVD movie recommendations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from recommender.model import SVDModel


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Exam" / "data" / "ml-latest-small-filtered"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
mcp = FastMCP("movielens-svd")


class UserMovieQuery(BaseModel):
    user_id: int
    movie_ids: list[int]


class Recommender:
    def __init__(self) -> None:
        model_path = ARTIFACT_DIR / "svd_model.npz"
        metadata_path = ARTIFACT_DIR / "svd_metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            raise RuntimeError("Model missing. Run: python -m recommender.train")
        self.model = SVDModel.load(str(model_path))
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ratings_path = DATA_DIR / "ratings.csv"
        if hashlib.sha256(ratings_path.read_bytes()).hexdigest() != self.metadata["ratings_sha256"]:
            raise RuntimeError("Ratings changed since training. Retrain the model.")
        self.ratings = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"])
        self.movies = pd.read_csv(DATA_DIR / "movies_with_plots.csv", usecols=["movieId", "title", "year", "genres"])
        self.movie_lookup = self.movies.set_index("movieId")
        self.rating_counts = self.ratings.groupby("movieId").size().to_dict()
        self.user_history = {
            int(user): group.sort_values(["rating", "movieId"], ascending=[False, True])
            for user, group in self.ratings.groupby("userId")
        }
        self.user_rating_maps = {
            user: dict(zip(group.movieId.astype(int), group.rating.astype(float)))
            for user, group in self.user_history.items()
        }

    def recommend(self, user_id: int, limit: int, genre: str | None, min_ratings: int) -> dict:
        if limit < 1 or limit > 50:
            raise ValueError("limit must be between 1 and 50")
        if min_ratings < 0:
            raise ValueError("min_ratings must be non-negative")
        if user_id not in self.user_history:
            raise ValueError(f"Unknown user_id: {user_id}")
        seen = set(self.user_history[user_id].movieId.tolist())
        scores = self.model.scores_for_user(user_id)
        candidates = pd.DataFrame({"movieId": self.model.movie_ids, "predicted_rating": scores})
        candidates = candidates[~candidates.movieId.isin(seen)]
        candidates["rating_count"] = candidates.movieId.map(self.rating_counts).fillna(0).astype(int)
        candidates = candidates[candidates.rating_count >= min_ratings]
        candidates = candidates.join(self.movie_lookup, on="movieId", how="inner")
        if genre:
            candidates = candidates[candidates.genres.str.split("|").map(lambda genres: genre.lower() in [g.lower() for g in genres])]
        top = candidates.sort_values(["predicted_rating", "rating_count", "movieId"], ascending=[False, False, True]).head(limit)
        liked = self.user_history[user_id].head(5).join(self.movie_lookup, on="movieId")
        return {
            "user_id": user_id,
            "ranking_method": "SVD predicted rating; excludes movies already rated by this user",
            "note": "Predictions are estimates, not explanations. liked_history contains observed ratings for grounded follow-up.",
            "filters": {"genre": genre, "min_ratings": min_ratings},
            "recommendations": [
                {
                    "movie_id": int(row.movieId),
                    "title": row.title,
                    "year": int(row.year),
                    "genres": row.genres.split("|"),
                    "predicted_rating": round(float(row.predicted_rating), 3),
                    "rating_count": int(row.rating_count),
                }
                for row in top.itertuples(index=False)
            ],
            "liked_history": [
                {"movie_id": int(row.movieId), "title": row.title, "rating": float(row.rating)}
                for row in liked.itertuples(index=False)
            ],
        }

    def similar_users(self, user_id: int, limit: int) -> dict:
        if limit < 1 or limit > 50:
            raise ValueError("limit must be between 1 and 50")
        matches = np.flatnonzero(self.model.user_ids == user_id)
        if not len(matches):
            raise ValueError(f"Unknown user_id: {user_id}")

        target_index = int(matches[0])
        factors = self.model.user_factors
        target = factors[target_index]
        norms = np.linalg.norm(factors, axis=1)
        denominator = norms * np.linalg.norm(target)
        similarities = np.divide(
            factors @ target,
            denominator,
            out=np.zeros(len(factors), dtype=float),
            where=denominator > 0,
        )
        neighbors = sorted(
            (
                (int(other_id), float(similarities[index]))
                for index, other_id in enumerate(self.model.user_ids)
                if index != target_index
            ),
            key=lambda pair: (-pair[1], pair[0]),
        )[:limit]
        target_ratings = self.user_rating_maps[user_id]
        results = []
        for other_id, similarity in neighbors:
            other_ratings = self.user_rating_maps[other_id]
            common = target_ratings.keys() & other_ratings.keys()
            mean_difference = (
                sum(abs(target_ratings[movie_id] - other_ratings[movie_id]) for movie_id in common) / len(common)
                if common else None
            )
            results.append({
                "user_id": other_id,
                "cosine_similarity": round(similarity, 4),
                "rating_count": len(other_ratings),
                "common_rated_movies": len(common),
                "mean_absolute_rating_difference_on_common": round(mean_difference, 3) if mean_difference is not None else None,
            })
        return {
            "user_id": user_id,
            "target_rating_count": len(target_ratings),
            "ranking_method": "cosine similarity of user factors learned by the full-data SVD model",
            "note": "Similarity is a model-based estimate. Common-rated counts and rating differences provide observed context; inspect neighbors' actual movie ratings before describing their opinion.",
            "similar_users": results,
        }

    def find_ratings(self, users: list[UserMovieQuery]) -> dict:
        if len(users) > 50 or sum(len(user.movie_ids) for user in users) > 1000:
            raise ValueError("Maximum 50 users and 1000 user/movie pairs per call")
        results = []
        for user in users:
            if user.user_id not in self.user_rating_maps:
                raise ValueError(f"Unknown user_id: {user.user_id}")
            user_ratings = self.user_rating_maps[user.user_id]
            movie_results = []
            for movie_id in user.movie_ids:
                if movie_id not in self.movie_lookup.index:
                    raise ValueError(f"Unknown movie_id: {movie_id}")
                movie = self.movie_lookup.loc[movie_id]
                movie_results.append({
                    "movie_id": movie_id,
                    "title": movie.title,
                    "year": int(movie.year),
                    "rating": user_ratings.get(movie_id),
                })
            results.append({
                "user_id": user.user_id,
                "rated_count": sum(movie["rating"] is not None for movie in movie_results),
                "movies": movie_results,
            })
        return {
            "note": "Ratings are observed values from ratings.csv; null means this user has not rated that movie.",
            "users": results,
        }


_recommender: Recommender | None = None


@mcp.tool()
def recommend_movies(user_id: int, limit: int = 10, genre: str | None = None, min_ratings: int = 5) -> dict:
    """Recommend unseen movies for a known MovieLens user using trained SVD ratings.

    genre is an exact MovieLens genre (case-insensitive), e.g. Thriller or Sci-Fi.
    Results include predicted scores, rating counts, and observed liked history.
    """
    global _recommender
    if _recommender is None:
        _recommender = Recommender()
    return _recommender.recommend(user_id, limit, genre, min_ratings)


@mcp.tool()
def find_similar_users(user_id: int, limit: int = 10) -> dict:
    """Return other users sorted by cosine similarity of trained SVD user factors.

    Includes observed rating overlap for context. This does not fetch neighbors'
    ratings for a particular movie.
    """
    global _recommender
    if _recommender is None:
        _recommender = Recommender()
    return _recommender.similar_users(user_id, limit)


@mcp.tool()
def find_ratings(users: list[UserMovieQuery]) -> dict:
    """Look up observed ratings for each requested user and their movie IDs.

    Pass users=[{"user_id": 1, "movie_ids": [296, 1]}, ...]. Results preserve
    input order; rating is null when a user has not rated a known movie.
    """
    global _recommender
    if _recommender is None:
        _recommender = Recommender()
    return _recommender.find_ratings(users)


if __name__ == "__main__":
    mcp.run(transport="stdio")
