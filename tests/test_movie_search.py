"""Exercise the search branch without live Gemini calls or trained artifacts."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import pandas as pd

from recommender.movie_search import MovieCatalog, TitleGuesses, normalize_title


class MovieSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / "movies.csv"
        pd.DataFrame([
            [1, "Toy Story", 1995, "Animation|Comedy", "A CSV-only toy story."],
            [2, "Interstellar", 2014, "Sci-Fi", "A CSV-only space story."],
            [3, "King Kong", 1933, "Adventure", "Original plot."],
            [4, "King Kong", 2005, "Adventure", "Remake plot."],
            [5, "Café", 2010, "Drama", ""],
        ], columns=["movieId", "title", "year", "genres", "plot"]).to_csv(path, index=False)
        self.catalog = MovieCatalog(path)

    async def test_exact_normalization_year_and_missing_plot_skip_llm(self):
        guesser = AsyncMock()
        response = await self.catalog.search([" TOY--STORY ", "King Kong (2005)", "Cafe\u0301"], guesser)
        self.assertEqual([item["status"] for item in response["results"]], ["exact"] * 3)
        self.assertEqual(response["results"][1]["movies"][0]["movie_id"], 4)
        self.assertIsNone(response["results"][2]["movies"][0]["plot"])
        guesser.assert_not_awaited()
        self.assertNotEqual(normalize_title("Toy Story 2"), normalize_title("Toy Story"))

    async def test_mixed_batch_preserves_exact_and_guesses_only_missing(self):
        guesser = AsyncMock(return_value={"guesses": [
            {"query_index": 1, "titles": ["Interstellar", "INTER STELLAR", "Imaginary movie"]},
            {"query_index": 2, "titles": ["Another imaginary movie"]},
        ]})
        results = (await self.catalog.search(["Toy Story", "Interstelar", "Missing"], guesser))["results"]
        guesser.assert_awaited_once_with([
            {"query_index": 1, "title": "Interstelar"}, {"query_index": 2, "title": "Missing"},
        ])
        self.assertEqual([item["status"] for item in results], ["exact", "guessed", "not_found"])
        self.assertEqual(results[1]["query_title"], "Interstelar")
        self.assertEqual(results[1]["match_method"], "guessed")
        self.assertEqual(len(results[1]["movies"]), 1)
        self.assertEqual(results[1]["movies"][0]["plot"], "A CSV-only space story.")

    async def test_exact_and_guessed_ambiguity(self):
        guesser = AsyncMock(return_value=TitleGuesses.model_validate({"guesses": [
            {"query_index": 1, "titles": ["King Kong", "King Kong (2005)"]},
        ]}))
        results = (await self.catalog.search(["King Kong", "King Kon"], guesser))["results"]
        self.assertEqual([item["status"] for item in results], ["ambiguous", "ambiguous"])
        self.assertEqual([item["match_method"] for item in results], ["exact", "guessed"])
        self.assertEqual(len(results[1]["movies"]), 2)

    async def test_failed_or_invalid_guessing_preserves_exact_results(self):
        bad_outputs = [
            {"guesses": [{"query_index": 1, "titles": ["Interstellar"] * 11}]},
            {"guesses": [{"query_index": 0, "titles": ["Interstellar"]}]},
            {"guesses": []},
            {"guesses": [{"query_index": 1, "titles": []}] * 2},
        ]
        for output in bad_outputs:
            with self.subTest(output=output):
                results = (await self.catalog.search(
                    ["Toy Story", "Interstelar"], AsyncMock(return_value=output),
                ))["results"]
                self.assertEqual(results[0]["status"], "exact")
                self.assertEqual(results[1]["status"], "lookup_error")
        results = (await self.catalog.search(
            ["Toy Story", "Interstelar"], AsyncMock(side_effect=RuntimeError("API unavailable")),
        ))["results"]
        self.assertEqual([item["status"] for item in results], ["exact", "lookup_error"])

    async def test_empty_candidates_are_not_found_and_inputs_are_validated(self):
        response = await self.catalog.search(["Unknown"], AsyncMock(return_value={
            "guesses": [{"query_index": 0, "titles": []}],
        }))
        self.assertEqual(response["results"][0]["status"], "not_found")
        for titles in ([], ["!!!"], [""], ["x"] * 21, "Toy Story", [None]):
            with self.subTest(titles=titles), self.assertRaises(ValueError):
                await self.catalog.search(titles)


if __name__ == "__main__":
    unittest.main()
