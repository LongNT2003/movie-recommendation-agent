"""Inspect held-out predictions or manually review unseen recommendations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from recommender.model import SVDModel
from recommender.server import DATA_DIR, ARTIFACT_DIR, Recommender
from recommender.train import split_per_user


PROJECT_DIR = Path(__file__).resolve().parent.parent


def print_history(history: pd.DataFrame, movie_lookup: pd.DataFrame) -> None:
    print("\nPhim đã chấm cao trong lịch sử:")
    for row in history.sort_values(["rating", "movieId"], ascending=[False, True]).head(5).itertuples(index=False):
        title = movie_lookup.loc[row.movieId, "title"]
        print(f"  {row.rating:.1f}/5  {title} (movieId={row.movieId})")


def inspect_holdout(user_id: int, limit: int) -> None:
    metadata = json.loads((ARTIFACT_DIR / "svd_metadata.json").read_text(encoding="utf-8"))
    ratings_path = DATA_DIR / "ratings.csv"
    if hashlib.sha256(ratings_path.read_bytes()).hexdigest() != metadata["ratings_sha256"]:
        raise RuntimeError("Ratings changed since training; retrain first")
    ratings = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"])
    train, validation = split_per_user(ratings, metadata["hyperparameters"]["seed"])
    if user_id not in set(ratings.userId):
        raise ValueError(f"Unknown user_id: {user_id}")
    model = SVDModel.load(str(ARTIFACT_DIR / "svd_validation_model.npz"))
    movies = pd.read_csv(DATA_DIR / "movies_with_plots.csv", usecols=["movieId", "title", "year", "genres"]).set_index("movieId")
    user_train = train[train.userId == user_id]
    user_validation = validation[validation.userId == user_id]
    print(f"User {user_id}: {len(user_train)} train ratings, {len(user_validation)} held out")
    print_history(user_train, movies)
    known = set(model.movie_ids.tolist())
    examples = []
    for row in user_validation.itertuples(index=False):
        predicted = model.predict(user_id, int(row.movieId)) if row.movieId in known else None
        examples.append((int(row.movieId), float(row.rating), predicted))
    covered = [entry for entry in examples if entry[2] is not None]
    if covered:
        mae = sum(abs(actual - predicted) for _, actual, predicted in covered) / len(covered)
        print(f"\nMAE của user trên rating giữ lại: {mae:.3f} ({len(covered)}/{len(examples)} phim có trong train)")
    print(f"Các dự đoán sai nhiều nhất (tối đa {limit}):")
    for movie_id, actual, predicted in sorted(examples, key=lambda x: abs(x[1] - x[2]) if x[2] is not None else -1, reverse=True)[:limit]:
        title = movies.loc[movie_id, "title"]
        year = int(movies.loc[movie_id, "year"])
        if predicted is None:
            print(f"  {title} ({year}, movieId={movie_id}): thực tế {actual:.1f}, chưa dự đoán được (phim không có trong train)")
        else:
            print(f"  {title} ({year}, movieId={movie_id}): dự đoán {predicted:.2f}, thực tế {actual:.1f}, sai lệch {abs(actual - predicted):.2f}")


def review_recommendations(user_id: int, limit: int, genre: str | None, min_ratings: int, no_prompt: bool, output: Path) -> None:
    recommender = Recommender()
    result = recommender.recommend(user_id, limit, genre, min_ratings)
    print(f"User {user_id}: {len(recommender.user_history[user_id])} ratings trong lịch sử")
    print_history(recommender.user_history[user_id], recommender.movie_lookup)
    print("\nGợi ý phim chưa được user chấm trong dữ liệu:")
    reviews = []
    for index, movie in enumerate(result["recommendations"], 1):
        print(f"\n{index}. {movie['title']} ({movie['year']}) | dự đoán {movie['predicted_rating']:.2f}/5 | {movie['rating_count']} ratings")
        print(f"   {', '.join(movie['genres'])} | movieId={movie['movie_id']}")
        if no_prompt:
            continue
        while True:
            answer = input("   Bạn đánh giá mức phù hợp 1–5 (Enter để bỏ qua): ").strip()
            if not answer:
                break
            if answer in {"1", "2", "3", "4", "5"}:
                note = input("   Ghi chú ngắn (có thể bỏ trống): ").strip()
                reviews.append({
                    "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "user_id": user_id,
                    "movie_id": movie["movie_id"],
                    "title": movie["title"],
                    "predicted_rating": movie["predicted_rating"],
                    "manual_relevance_1_to_5": int(answer),
                    "note": note,
                })
                break
            print("   Hãy nhập 1–5 hoặc Enter.")
    if reviews:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(reviews[0]))
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerows(reviews)
        print(f"\nĐã lưu {len(reviews)} đánh giá vào {output}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--mode", choices=["holdout", "review"], default="holdout")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--genre", type=str)
    parser.add_argument("--min-ratings", type=int, default=5)
    parser.add_argument("--no-prompt", action="store_true", help="Display recommendations without collecting manual scores")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "evaluations" / "manual_reviews.csv")
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 50:
        parser.error("--limit must be between 1 and 50")
    if args.mode == "holdout":
        inspect_holdout(args.user_id, args.limit)
    else:
        review_recommendations(args.user_id, args.limit, args.genre, args.min_ratings, args.no_prompt, args.output)


if __name__ == "__main__":
    main()
