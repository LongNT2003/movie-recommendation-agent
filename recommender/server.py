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
from recommender.movie_search import MovieCatalog
from recommender.trace import log_event, log_exception


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

    def recommend(self, user_id: int, limit: int, genre: str | None, min_ratings: int,
                  include_genres: list[str] | None = None,
                  exclude_genres: list[str] | None = None) -> dict:
        if limit < 1 or limit > 50:
            raise ValueError("limit must be between 1 and 50")
        if min_ratings < 0:
            raise ValueError("min_ratings must be non-negative")
        if user_id not in self.user_history:
            raise ValueError(f"Unknown user_id: {user_id}")
        known_genres = {value.casefold(): value for genres in self.movie_lookup.genres
                        for value in genres.split("|")}

        def normalize_genres(values, field):
            if values is None:
                return []
            if not isinstance(values, list):
                raise ValueError(f"{field} must be a list of genre names")
            normalized = []
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{field} must contain non-empty genre names")
                key = value.strip().casefold()
                if key not in known_genres:
                    raise ValueError(f"Unknown genre: {value}. Valid genres: {', '.join(sorted(known_genres.values()))}")
                canonical = known_genres[key]
                if canonical not in normalized:
                    normalized.append(canonical)
            return normalized

        included = normalize_genres(include_genres, "include_genres")
        # Keep the original single-genre argument as an alias for inclusion.
        if genre is not None:
            included = normalize_genres([*included, genre], "genre")
        excluded = normalize_genres(exclude_genres, "exclude_genres")
        overlap = set(included) & set(excluded)
        if overlap:
            raise ValueError(f"Genres cannot be both included and excluded: {', '.join(sorted(overlap))}")
        seen = set(self.user_history[user_id].movieId.tolist())
        scores = self.model.scores_for_user(user_id)
        candidates = pd.DataFrame({"movieId": self.model.movie_ids, "predicted_rating": scores})
        candidates = candidates[~candidates.movieId.isin(seen)]
        candidates["rating_count"] = candidates.movieId.map(self.rating_counts).fillna(0).astype(int)
        candidates = candidates[candidates.rating_count >= min_ratings]
        candidates = candidates.join(self.movie_lookup, on="movieId", how="inner").reset_index(drop=True)
        if included or excluded:
            include_set, exclude_set = set(included), set(excluded)
            candidates = candidates[candidates.genres.str.split("|").map(
                lambda genres: (not include_set or bool(include_set.intersection(genres)))
                and not exclude_set.intersection(genres)
            ).astype(bool)]
        available_count = len(candidates)
        top = candidates.sort_values(["predicted_rating", "rating_count", "movieId"], ascending=[False, False, True]).head(limit)
        liked = self.user_history[user_id].head(5).join(self.movie_lookup, on="movieId")
        return {
            "user_id": user_id,
            "ranking_method": "SVD predicted rating; excludes movies already rated by this user",
            "note": (
                "Predictions are estimates, not explanations. liked_history contains observed ratings for grounded follow-up. "
                "Genre constraints are hard filters applied before selecting the top results. "
                "Chat preferences do not retrain the SVD model."
                + (" Fewer movies satisfy the filters than requested; filters were not relaxed."
                   if available_count < limit else "")
            ),
            "filters": {"genre": genre, "include_genres": included,
                        "exclude_genres": excluded, "min_ratings": min_ratings},
            "requested_limit": limit,
            "returned_count": len(top),
            "available_count": available_count,
            "status": "ok" if available_count >= limit else "insufficient_candidates",
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
        trained_movie_ids = set(self.model.movie_ids.tolist())
        for user in users:
            if user.user_id not in self.user_rating_maps:
                raise ValueError(f"Unknown user_id: {user.user_id}")
            user_ratings = self.user_rating_maps[user.user_id]
            movie_results = []
            for movie_id in user.movie_ids:
                if movie_id not in self.movie_lookup.index:
                    raise ValueError(f"Unknown movie_id: {movie_id}")
                movie = self.movie_lookup.loc[movie_id]
                observed_rating = user_ratings.get(movie_id)
                if observed_rating is not None:
                    rating = float(observed_rating)
                    rating_source = "observed"
                elif movie_id in trained_movie_ids:
                    rating = round(self.model.predict(user.user_id, movie_id), 3)
                    rating_source = "predicted"
                else:
                    rating = None
                    rating_source = "unavailable"
                result = {
                    "movie_id": movie_id,
                    "title": movie.title,
                    "year": int(movie.year),
                    "rating": rating,
                    "rating_source": rating_source,
                }
                if rating_source == "unavailable":
                    result["reason"] = "Movie has no training ratings; SVD cannot predict its score."
                movie_results.append(result)
            results.append({
                "user_id": user.user_id,
                "rated_count": sum(movie["rating_source"] == "observed" for movie in movie_results),
                "predicted_count": sum(movie["rating_source"] == "predicted" for movie in movie_results),
                "unavailable_count": sum(movie["rating_source"] == "unavailable" for movie in movie_results),
                "movies": movie_results,
            })
        return {
            "note": (
                "rating_source=observed means the user actually rated the movie in ratings.csv. "
                "rating_source=predicted means the user has not rated it and rating is an SVD "
                "estimate on the 0.5–5 scale, not an observed rating. rating_source=unavailable "
                "has rating=null because the movie has no training ratings. rated_count counts "
                "only observed ratings. Always distinguish observed ratings from predictions."
            ),
            "users": results,
        }


_recommender: Recommender | None = None
_movie_catalog: MovieCatalog | None = None


@mcp.tool()
async def get_movie_info(titles: list[str]) -> dict:
    """Get CSV movie details, including plots, for 1 to 20 user-supplied titles.

    Searches normalized exact titles first. For misses, internally asks Gemini for
    up to 10 corrected titles per input in one pass, then searches exact again.
    Status is exact, guessed, ambiguous, not_found or lookup_error per input.
    Disclose guessed matches; ask the user to select ambiguous matches; report misses.
    Pass original titles, not agent-generated guesses. No trained model is required.
    """
    global _movie_catalog
    if _movie_catalog is None:
        csv_path = DATA_DIR / "movies_with_plots.csv"
        log_event("search.csv.load.start", path=str(csv_path))
        try:
            _movie_catalog = MovieCatalog(csv_path)
        except Exception as exc:
            log_exception("search.csv.load.error", exc)
            raise
        log_event("search.csv.load.end", path=str(csv_path))
    return await _movie_catalog.search(titles)


@mcp.tool()
def recommend_movies(user_id: int, limit: int = 10, genre: str | None = None, min_ratings: int = 5,
                     include_genres: list[str] | None = None,
                     exclude_genres: list[str] | None = None) -> dict:
    """Recommend unseen movies for a known MovieLens user using trained SVD ratings.

    Reuse the known user ID from history or session memory. include_genres requires
    at least one listed genre; empty/omitted means unrestricted. exclude_genres
    removes movies containing ANY listed genre, including mixed-genre movies.
    Use exclude_genres=["Animation"] for "tired of animated movies". Genres are
    case-insensitive MovieLens names, e.g. Thriller or Sci-Fi. Overlap is an error.
    genre is a legacy single-genre alias merged into include_genres (OR matching).
    All filters run before top-limit selection and are never relaxed automatically.
    Results include predicted scores, applied filters, counts and observed history.
    insufficient_candidates means fewer eligible movies than requested.
    """
    global _recommender
    if _recommender is None:
        _recommender = Recommender()
    return _recommender.recommend(user_id, limit, genre, min_ratings, include_genres, exclude_genres)


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
    """Get actual ratings, or SVD predictions for movies a user has not rated.

    Pass users=[{"user_id": 1, "movie_ids": [296, 1]}, ...]. Results preserve
    input order. Each rating_source is observed (actually rated), predicted (SVD
    estimate), or unavailable (no training ratings for the movie, rating=null).
    Use this to assess whether a user might like a specific movie. Disclose predictions
    as estimates. rated_count includes observed ratings only. Use database movie IDs.
    """
    global _recommender
    if _recommender is None:
        _recommender = Recommender()
    return _recommender.find_ratings(users)


if __name__ == "__main__":
    mcp.run(transport="stdio")
