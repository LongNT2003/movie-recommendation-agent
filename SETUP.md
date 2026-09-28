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
- `CHAT_HISTORY_MAX_MESSAGES` controls how many previous user messages and final assistant answers are retained (default: `10`; `0` sends none). Intermediate assistant tool calls and their tool results are included with retained turns and do not count against this limit. A cutoff inside a turn containing tools expands to keep the entire turn, so no tool call/result pairs are split. The current query is always included. The limit affects context sent to the LLM; it does not delete entries from the session history.

### Movie title lookup

`get_movie_info(titles=["Toy Story", "Interstelar"])` accepts 1–20 titles and reads
`Exam/data/ml-latest-small-filtered/movies_with_plots.csv`, independently of trained
SVD artifacts. It matches exact titles after Unicode NFKC normalization, case folding,
and removal of whitespace and punctuation. Letters and numbers are retained. A year
can be supplied, for example `King Kong (1933)`, to disambiguate remakes.

Only unresolved titles enter one internal Gemini call, using the same `GOOGLE_API_KEY`
and `GEMINI_MODEL` as the agent. Gemini suggests at most 10 candidate titles per input;
these candidates are searched exactly against the CSV. Generated movie facts are never
returned. All movie details, including `plot`, come from matching CSV rows.

The response preserves input order in `results`, with `query_title`, `query_index`,
`status`, `match_method`, `attempted_titles`, and `movies` for each input. Status is
`exact`, `guessed`, `ambiguous`, `not_found`, or `lookup_error`. The agent discloses title
corrections, asks the user to choose ambiguous matches, and reports unresolved titles.
If correction fails, exact matches are preserved and unresolved inputs are marked
`lookup_error` rather than being reported as confirmed misses. Exact-only calls need
no API key. Searching plots by semantic description is not part of this tool.

### Chat logs

Each agent session writes two UTF-8 files with the same name:
`logs/chat-<UTC timestamp>-<id>.log` is a readable transcript; `.jsonl` contains
the same events as compact JSON for scripts. The CLI prints the `.log` path.

Logs contain `USER INPUT`, `TOOL USE` (name and arguments), `TOOL RESULT` (full
content, including plots and rating sources), and `LLM OUTPUT` (final answer).
Each tool execution is recorded once, including memory tools. Turn and call IDs
connect parallel tool executions to their results. Tool JSON is decoded and indented
in the readable file; message wrappers and duplicate MCP artifacts are omitted.
System prompts, history snapshots, token metadata and internal search steps are
omitted. Errors retain the failing step, exception type and message without a
stack trace. Known secrets and credential fields are redacted in both files.
`logs/` is ignored by Git. Existing log files are unchanged.

To follow a session live in another PowerShell terminal, use the path printed by CLI:

```powershell
Get-Content -LiteralPath 'logs/chat-<timestamp>-<id>.log' -Encoding UTF8 -Wait
```

For library use, pass `MovieAgent(trace_path=Path("logs/debug.jsonl"))` to choose a
JSONL file; the readable counterpart is available as `agent.trace.readable_path`.
No additional environment variables are needed; `MOVIE_AGENT_TRACE_FILE` is
passed internally to the MCP subprocess.

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
# history now also contains this query, assistant tool calls, tool results, and answer.
```

Reuse the same history list on each call. `chat(query, history)` appends the complete successful turn to that list automatically: the user message, all assistant tool calls, every `ToolMessage`, and the final assistant answer. Do not append the user/assistant pair yourself. Failed turns leave history unchanged. Existing dictionaries with `role: user/assistant` are still accepted; new entries are native LangChain messages preserving tool call IDs, content blocks, names, and error statuses. Dictionaries with `role: tool` and matching `tool_call_id` are also supported. System prompts are regenerated each turn and are not stored in history. Full history stays in process memory for the session and is not persisted across CLI restarts.

Pass the same `SessionMemory` object for subsequent turns in one chat session; create a new object for a new session. Calling `chat(query, history)` without one creates memory for that call only. For many turns in one process, use `async with MovieAgent() as agent:` and call `agent.chat(query, history)` to reuse both the MCP connection and session memory. Retained tool messages let the agent reuse verified movie IDs and previously fetched facts in follow-up questions.

`recommend_movies(user_id, limit=10, genre=None, min_ratings=5)` returns unseen movies ranked by predicted rating, with genres, number of observed ratings, and the user's top observed ratings. It does not claim that these examples caused the prediction. Movies without any training rating cannot be scored by this SVD model. The default minimum of 5 ratings reduces the weakest item estimates; pass `min_ratings=0` to include rated but very sparse movies.

`find_similar_users(user_id, limit=10)` returns other users sorted by cosine similarity of the user factors learned by the final SVD model. Each result includes its rating count, number of movies rated by both users, and average absolute rating difference on those shared movies (or `null` if none). The ranking uses latent cosine similarity; it does not group users into clusters or report their ratings for a specified movie.

`find_ratings(users)` returns actual ratings when present, otherwise SVD predictions for movies a user has not rated. Example input: `users=[{"user_id": 476, "movie_ids": [296, 1]}, {"user_id": 290, "movie_ids": [296]}]`. The result preserves user/movie order. Each movie has `rating` and `rating_source`: `observed` means an actual rating from `ratings.csv`; `predicted` means an SVD estimate, rounded to 3 decimal places on the 0.5–5 scale; `unavailable` means the movie has no training ratings, with `rating: null` and a reason. `rated_count` counts only observed entries; `predicted_count` and `unavailable_count` count the other entries. The agent must explicitly distinguish actual ratings from estimates. It rejects unknown IDs and limits one call to 50 users / 1000 user-movie pairs.

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
