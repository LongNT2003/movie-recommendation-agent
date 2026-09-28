"""Check genre constraints against controlled SVD rankings without training."""

import unittest

import numpy as np
import pandas as pd

from recommender.model import SVDModel
from recommender.server import Recommender


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.recommender = Recommender.__new__(Recommender)
        self.recommender.model = SVDModel(
            user_ids=np.array([1]), movie_ids=np.arange(10, 17),
            user_factors=np.zeros((1, 1)), movie_factors=np.zeros((7, 1)),
            user_bias=np.zeros(1), movie_bias=np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
            global_mean=3.0,
        )
        self.recommender.movie_lookup = pd.DataFrame([
            [10, "Seen", 1995, "Comedy"],
            [11, "Animated comedy", 1996, "Animation|Comedy"],
            [12, "Animated drama", 1997, "Drama|Animation"],
            [13, "Live comedy", 1998, "Comedy"],
            [14, "Live drama", 1999, "Drama"],
            [15, "Live thriller", 2000, "Drama|Thriller"],
            [16, "Sparse comedy", 2001, "Comedy"],
        ], columns=["movieId", "title", "year", "genres"]).set_index("movieId")
        self.recommender.rating_counts = {movie_id: 10 for movie_id in range(10, 17)}
        self.recommender.rating_counts[16] = 1
        self.recommender.user_history = {1: pd.DataFrame([[10, 4.5]], columns=["movieId", "rating"])}

    def recommend(self, **kwargs):
        return self.recommender.recommend(1, kwargs.pop("limit", 2), kwargs.pop("genre", None),
                                          kwargs.pop("min_ratings", 5), **kwargs)

    def test_exclusion_runs_before_top_limit_and_removes_mixed_genres(self):
        result = self.recommend(exclude_genres=["Animation"])
        self.assertEqual([movie["movie_id"] for movie in result["recommendations"]], [13, 14])
        self.assertEqual(result["available_count"], 3)
        self.assertEqual(result["returned_count"], 2)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["liked_history"][0]["movie_id"], 10)

    def test_inclusion_is_or_exclusion_is_any_and_genres_are_normalized(self):
        result = self.recommend(include_genres=[" comedy ", "DRAMA", "Comedy"],
                                exclude_genres=[" animation ", "THRILLER", "Animation"])
        self.assertEqual([movie["movie_id"] for movie in result["recommendations"]], [13, 14])
        self.assertEqual(result["filters"]["include_genres"], ["Comedy", "Drama"])
        self.assertEqual(result["filters"]["exclude_genres"], ["Animation", "Thriller"])

    def test_legacy_genre_and_unrestricted_calls_keep_existing_rankings(self):
        self.assertEqual([m["movie_id"] for m in self.recommend()["recommendations"]], [11, 12])
        self.assertEqual([m["movie_id"] for m in self.recommend(genre="comedy")["recommendations"]], [11, 13])
        result = self.recommend(genre="Comedy", include_genres=["Drama"], exclude_genres=["Animation"])
        self.assertEqual(result["filters"]["include_genres"], ["Drama", "Comedy"])
        self.assertEqual([m["movie_id"] for m in result["recommendations"]], [13, 14])

    def test_shortage_and_empty_results_do_not_relax_constraints(self):
        result = self.recommend(limit=5, include_genres=["Comedy"], exclude_genres=["Animation"])
        self.assertEqual([m["movie_id"] for m in result["recommendations"]], [13])
        self.assertEqual(result["status"], "insufficient_candidates")
        self.assertEqual(result["requested_limit"], 5)
        self.assertEqual(result["returned_count"], 1)
        self.assertIn("not relaxed", result["note"])
        for filters in ({"include_genres": ["Animation"], "exclude_genres": ["Comedy", "Drama"]},
                        {"exclude_genres": ["Animation"], "min_ratings": 100}):
            with self.subTest(filters=filters):
                empty = self.recommend(**filters)
                self.assertEqual(empty["recommendations"], [])
                self.assertEqual(empty["available_count"], 0)
                self.assertEqual(empty["status"], "insufficient_candidates")

    def test_invalid_unknown_and_conflicting_genres_are_rejected(self):
        for filters in (
            {"include_genres": ["animation"], "exclude_genres": [" Animation "]},
            {"genre": "Comedy", "exclude_genres": ["comedy"]},
            {"include_genres": ["Invented genre"]}, {"exclude_genres": [""]},
            {"exclude_genres": [123]}, {"include_genres": "Comedy"},
        ):
            with self.subTest(filters=filters), self.assertRaises(ValueError):
                self.recommend(**filters)


if __name__ == "__main__":
    unittest.main()
