# SVD recommender MCP tool

This implementation trains one biased matrix factorization model (Funk SVD) on explicit ratings. It exposes MCP tools over stdio and includes a Gemini chat agent built with LangGraph.

## Setup (PowerShell, Python 3.11+)

Run from the project root:

```powershell
./scripts/setup.ps1
```

Or run the steps individually:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m recommender.train
```

The training command runs a reproducible 10% per-user holdout, prints RMSE/MAE against a user-mean baseline plus Hit Rate@10 and Recall@10 against a smoothed movie-mean baseline, then refits on all 74,064 ratings. It saves `artifacts/svd_model.npz`, `artifacts/svd_validation_model.npz`, and `artifacts/svd_metadata.json`. Change model settings with `--factors`, `--epochs`, `--learning-rate`, `--regularization`, and `--seed`.

The measured results and limitations are in `TRAIN_EVALUATION.md`.

To measure how closely the final 100%-trained artifact fits its own training ratings, run `.venv\Scripts\python.exe -m scripts.evaluate_full`. This writes `artifacts/full_train_evaluation.json`. Because the model has already seen every row, these in-sample metrics are not a test of recommendation quality.

## Start the MCP server

```powershell
.venv\Scripts\python.exe -m recommender.server
```

Configure an MCP client with stdio command `D:\Projects\trustedAI-project\.venv\Scripts\python.exe`, arguments `-m`, `recommender.server`, and working directory `D:\Projects\trustedAI-project`.

## Run the chat agent

Create `.env` in the project root with these values:

```dotenv
GOOGLE_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-3.8-flash
CHAT_HISTORY_MAX_MESSAGES=10
```

Only `GOOGLE_API_KEY` is required. The other two entries show the defaults and can be omitted. `.env` is ignored by Git; environment variables already set in the terminal take precedence. Start the interactive CLI:

```powershell
.venv\Scripts\python.exe -m recommender.chat_cli
```

Use `/exit` to quit. The agent discovers every tool exposed by the local MCP server when it starts and adds `read_session_memory` and `write_session_memory`. Gemini can save important facts, such as a MovieLens user ID, in memory for this CLI session. The memory is included in the system prompt on every turn, regardless of the chat history limit. It is kept only in process memory and disappears when the CLI exits. The agent needs the trained files in `artifacts/` before calling recommendation tools.

Optional settings:

- `GEMINI_MODEL` selects the Gemini model (default: `gemini-3.8-flash`).
- `CHAT_HISTORY_MAX_MESSAGES` controls how many previous user and assistant messages are sent per turn (default: `10`; `0` sends none). The current query is always included.

Python callers can supply their own conversation history:

```python
import asyncio
from recommender.agent import SessionMemory, chat

session_memory = SessionMemory()
history = [
    {"role": "user", "content": "Gợi ý phim cho user 1"},
    {"role": "assistant", "content": "..."},
]
answer = asyncio.run(chat("Có phim hành động nào?", history, session_memory=session_memory))
print(answer)
```

The caller stores and supplies history on each call. Pass the same `SessionMemory` object for subsequent turns in one chat session; create a new object for a new session. Calling `chat(query, history)` without one creates memory for that call only. For many turns in one process, use `async with MovieAgent() as agent:` and call `agent.chat(query, history)` to reuse both the MCP connection and session memory.

`recommend_movies(user_id, limit=10, genre=None, min_ratings=5)` returns unseen movies ranked by predicted rating, with genres, number of observed ratings, and the user's top observed ratings. It does not claim that these examples caused the prediction. Movies without any training rating cannot be scored by this SVD model. The default minimum of 5 ratings reduces the weakest item estimates; pass `min_ratings=0` to include rated but very sparse movies.

`find_similar_users(user_id, limit=10)` returns other users sorted by cosine similarity of the user factors learned by the final SVD model. Each result includes its rating count, number of movies rated by both users, and average absolute rating difference on those shared movies (or `null` if none). The ranking uses latent cosine similarity; it does not group users into clusters or report their ratings for a specified movie.

`find_ratings(users)` looks up actual ratings for multiple users and movie lists. Example input: `users=[{"user_id": 476, "movie_ids": [296, 1]}, {"user_id": 290, "movie_ids": [296]}]`. The result preserves user/movie order and includes `rating: null` if a user has not rated a known movie. It rejects unknown IDs and limits one call to 50 users / 1000 user-movie pairs.

Example: `recommend_movies(user_id=1, limit=3, genre="Action")` returns *North by Northwest*, *Yojimbo*, and *Kelly's Heroes* with the current default model settings.

## Manual evaluation

Inspect held-out ratings that the validation model did not train on, sorted by largest error:

```powershell
.venv\Scripts\python.exe -m scripts.manual_evaluate --mode holdout --user-id 1
```

Review new recommendations for a user and score their relevance from 1 to 5. Scores and notes are appended to `evaluations/manual_reviews.csv`:

```powershell
.venv\Scripts\python.exe -m scripts.manual_evaluate --mode review --user-id 15 --limit 5
```

Try `--user-id 30` for a sparse user history, or add `--genre Thriller` and `--min-ratings 10`. Use `--no-prompt` to print recommendations without saving reviews. The held-out mode shows actual ratings and prediction errors; the review mode shows genuinely unrated movies, so your 1–5 response is a subjective relevance judgment, not a measured rating prediction error.

Validation uses a random per-user holdout, not a chronological split. RMSE/MAE measure rating prediction, while Hit Rate@10 and Recall@10 use held-out ratings ≥4 as relevant and rank train-known films with at least 5 train ratings that are not in that user's train history. This matches the MCP tool's default `min_ratings=5` filter. Offline metrics cannot measure conversational usefulness; review outputs and failure cases manually.
