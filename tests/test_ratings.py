"""Verify actual-rating precedence and SVD fallback without training a model."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from recommender.model import SVDModel
from recommender.server import Recommender, UserMovieQuery


class RatingTests(unittest.TestCase):
    def setUp(self):
        self.recommender = Recommender.__new__(Recommender)
        self.recommender.model = SVDModel(
            user_ids=np.array([1, 2]), movie_ids=np.array([10, 20]),
            user_factors=np.array([[1.0], [-1.0]]),
            movie_factors=np.array([[0.5], [0.25]]),
            user_bias=np.array([0.0, 0.0]), movie_bias=np.array([0.0, 0.0]),
            global_mean=3.0,
        )
        self.recommender.user_rating_maps = {1: {10: 4.5}, 2: {20: 2.0}}
        self.recommender.movie_lookup = pd.DataFrame([
            [10, "Rated movie", 1995], [20, "Another movie", 2000],
            [30, "Untrained movie", 2010],
        ], columns=["movieId", "title", "year"]).set_index("movieId")

    def test_observed_rating_takes_precedence_and_does_not_call_predict(self):
        with patch.object(self.recommender.model, "predict", side_effect=AssertionError("Must not predict")):
            result = self.recommender.find_ratings([UserMovieQuery(user_id=1, movie_ids=[10])])["users"][0]
        self.assertEqual(result["movies"][0]["rating"], 4.5)
        self.assertEqual(result["movies"][0]["rating_source"], "observed")
        self.assertEqual(result["rated_count"], 1)
        self.assertEqual(result["predicted_count"], 0)

    def test_mixed_sources_preserve_input_order_and_counts(self):
        results = self.recommender.find_ratings([
            UserMovieQuery(user_id=1, movie_ids=[20, 10, 30]),
            UserMovieQuery(user_id=2, movie_ids=[10, 20]),
        ])["users"]
        self.assertEqual([user["user_id"] for user in results], [1, 2])
        first = results[0]
        self.assertEqual([movie["movie_id"] for movie in first["movies"]], [20, 10, 30])
        self.assertEqual([movie["rating_source"] for movie in first["movies"]],
                         ["predicted", "observed", "unavailable"])
        self.assertEqual([movie["rating"] for movie in first["movies"]], [3.25, 4.5, None])
        self.assertEqual((first["rated_count"], first["predicted_count"], first["unavailable_count"]), (1, 1, 1))
        self.assertIn("no training ratings", first["movies"][2]["reason"])
        self.assertEqual(results[1]["movies"][0]["rating"], 2.5)
        self.assertEqual(results[1]["movies"][1]["rating"], 2.0)

    def test_predictions_stay_within_rating_scale(self):
        self.recommender.model.global_mean = 20.0
        result = self.recommender.find_ratings([UserMovieQuery(user_id=1, movie_ids=[20])])
        self.assertEqual(result["users"][0]["movies"][0]["rating"], 5.0)
        self.recommender.model.global_mean = -20.0
        result = self.recommender.find_ratings([UserMovieQuery(user_id=1, movie_ids=[20])])
        self.assertEqual(result["users"][0]["movies"][0]["rating"], 0.5)

    def test_unknown_ids_and_pair_limits_still_rejected(self):
        for users in (
            [UserMovieQuery(user_id=999, movie_ids=[10])],
            [UserMovieQuery(user_id=1, movie_ids=[999])],
            [UserMovieQuery(user_id=1, movie_ids=[10])] * 51,
            [UserMovieQuery(user_id=1, movie_ids=[10] * 1001)],
        ):
            with self.subTest(users=users), self.assertRaises(ValueError):
                self.recommender.find_ratings(users)


if __name__ == "__main__":
    unittest.main()
