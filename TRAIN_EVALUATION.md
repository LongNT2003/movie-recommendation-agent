# SVD training evaluation

Run: `python -m recommender.train` with 32 factors, 20 epochs, learning rate 0.01, regularization 0.05, seed 42. The model was trained on `Exam/data/ml-latest-small-filtered/ratings.csv` (74,064 ratings, 610 users, 5,126 rated movies).

## Fit of the final 100%-trained model

`python -m scripts.evaluate_full` loads `artifacts/svd_model.npz` and predicts the same 74,064 ratings that were used to fit it. The output is saved in `artifacts/full_train_evaluation.json`.

| Training-set metric | SVD | User-mean baseline |
|---|---:|---:|
| RMSE | 0.6764 | 0.9384 |
| MAE | 0.5269 | 0.7331 |

On the 5,018 rating rows for movies with fewer than 5 ratings, SVD has RMSE 0.6847 and MAE 0.5391. On the other 69,046 rows, it has RMSE 0.6758 and MAE 0.5260. These are **in-sample** figures: the saved model has seen every rating being scored. They show training fit only and are expected to look better than held-out results. A top-10 hit/recall score cannot be computed honestly from this model and these same ratings while also excluding films the user has already rated.

## Protocol

Hold out 10% of each user's ratings at random (7,413 total), fit a validation model on the remaining 66,651, evaluate, then fit the production artifact on all 74,064 ratings. Rating prediction metrics cover 7,272 held-out rows whose movies are also in validation training; 141 held-out rows are cold items for that model. The random split is reproducible but does not simulate future ratings chronologically.

For top-10 metrics, a relevant film has a held-out rating of at least 4.0. Both SVD and the comparison rank films absent from the user's training history and with at least 5 training ratings, matching the MCP tool's default support filter. The ranking baseline is a movie mean smoothed with 10 global-mean pseudo-ratings.

| Metric | SVD | Baseline |
|---|---:|---:|
| RMSE (baseline: user's training mean) | 0.8471 | 0.9479 |
| MAE (baseline: user's training mean) | 0.6478 | 0.7356 |
| Hit Rate@10 (baseline: smoothed movie mean) | 0.1490 | 0.3555 |
| Recall@10 (baseline: smoothed movie mean) | 0.0339 | 0.0874 |

The ranking metrics include 557 users with at least one eligible relevant holdout and 3,274 eligible relevant ratings. SVD improves rating prediction over a user-mean baseline (RMSE about 10.6% lower), but its top-10 ranking is substantially worse than the smoothed movie-mean baseline. It should not yet be presented as a strong movie discovery ranker. One plausible explanation is that the highest SVD scores favor less-supported movies; the current ranking uses predicted rating alone. That explanation needs a targeted check before changing the ranker.

## Concrete failure

For user 1, the validation model predicted *Psycho* (1960, movieId 1219) at 4.82, while the held-out rating is 2.0 (absolute error 2.82). This is a case where a high model score would produce a poor recommendation for this user. Run `python -m scripts.manual_evaluate --mode holdout --user-id 1` to inspect the user context and other errors.

## Limits

These offline numbers do not show whether a person finds the recommendations useful, and unrated movies are not verified dislikes. The manual review script collects subjective relevance scores for genuinely unrated recommendations separately from held-out rating errors.
