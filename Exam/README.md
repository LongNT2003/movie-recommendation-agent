# TrustedAI - AI Engineer Test

## The Problem

You have a dataset of movies with plot summaries, user ratings, and tags. Your task:

**Build an AI assistant that helps users discover movies by investigating the dataset on their behalf.**

This is not a search engine — the assistant should reason about what to look up, combine multiple data signals, and explain its thinking. When a user asks "why would I like that?", the assistant should be able to dig into their rating history, find patterns, and give a grounded answer.

### Requirements

1. The user identifies themselves (e.g., by user ID), and the assistant uses their rating history to personalize recommendations
2. The assistant can answer questions that require combining multiple pieces of information — e.g., "what do people with similar taste to mine think of Inception?" requires finding similar users, checking their Inception ratings, and synthesizing
3. The assistant explains its reasoning using historical data — not just LLM knowledge
4. Evaluate your system's recommendation quality with evidence — show where it works and where it fails (you might use metrics, qualitative examples, or both — explain why you chose what you chose)

How you get there is up to you.

## Dataset

Located in `data/ml-latest-small-filtered/`:

| File | Size | Description |
|------|------|-------------|
| `movies_with_plots.csv` | 16MB | 5,135 movies — `movieId`, `title`, `year`, `genres`, `plot` (100–5,000+ chars, avg ~3,200) |
| `ratings.csv` | 1.7MB | 74,064 ratings from 610 users — `userId`, `movieId`, `rating` (0.5–5.0), `timestamp` |
| `tags.csv` | 74KB | 2,440 user-generated tags — sparse, most movies have none |
| `links.csv` | 98KB | External links (IMDb, TMDB) |
| `movies.csv` | 228KB | Basic movie info without plots |

**Data notes:**
- Movies span 1903–2014. This is a filtered subset of MovieLens — some well-known movies (e.g., The Matrix, Ocean's Eleven) may be absent due to missing plot data in the source.
- ~51% of movies have fewer than 5 ratings.
- Tags are very sparse (2,440 tags across 5,135 movies). Don't build your approach around tags alone.
- Users have ~121 ratings on average — relatively dense on the user side. The sparsity is on the movie side.

### What's in the data

The dataset gives you several signals to work with:

- **Rating patterns:** 610 users × 5,135 movies. Users who rate similar movies similarly have similar taste — this is the basis of collaborative filtering. You can find "users like me" and see what they enjoyed.
- **Plot summaries:** Full text descriptions (avg ~3,200 chars). Useful for content-based search — finding movies that match a description like "dark thriller with a twist."
- **Genres:** 19 genres per movie (pipe-separated). Useful for filtering, profiling user preferences, and finding blind spots.
- **Tags:** User-generated labels like "twist ending", "atmospheric", "dark comedy". Sparse but high-signal where they exist.

### Suggested users for testing

These users have different profiles — useful for testing personalization:

| User ID | Ratings | Avg | Profile |
|---------|---------|-----|---------|
| 1 | 190 | 4.33 | Action/comedy fan — likes Terminator, Blues Brothers, Full Metal Jacket |
| 15 | 85 | 3.55 | Sci-fi oriented — likes Aliens, Star Wars, Back to the Future |
| 30 | 18 | 4.61 | Sparse history — likes Braveheart, Inception, Shawshank Redemption |

### Sample Queries

Use these to sanity-check your system during development:

- "What should I watch tonight?" *(requires knowing the user's taste)*
- "I want a dark psychological thriller with a twist" *(content search + quality filter)*
- "What do people with similar taste to mine think about Pulp Fiction?" *(find similar users + aggregate their ratings)*
- "Why do you think I'd like that?" *(explain using user's history + movie data)*
- "I liked Toy Story but I'm tired of animated movies — what else?" *(use history + apply constraints)*
- "What's my blind spot? What genres am I missing?" *(analyze user's rating patterns)*

## Deliverables

### 1. Code
- Working implementation with setup instructions
- Include reproducible output: sample conversations, evaluation results, or screenshots that demonstrate the system working
- If your solution uses external APIs (e.g., OpenAI), document this and include example outputs so we can evaluate without running it

### 2. Report
Follow the template in `REPORT_TEMPLATE.md`. **This is as important as the code.**

We weight the report equally with the code. A mediocre system with excellent analysis beats a good system with a shallow report.

### 3. Interview
You will:
- Demo your system live
- Walk us through your report
- Discuss your decisions and tradeoffs

## Time

3-4 hours. Rough guide: ~2 hours building, ~1 hour on the report and evaluation, ~30 min cleanup.

Tip: keep notes on your decisions as you go — it makes the report much easier to write.

AI tools are welcome — be ready to discuss your work in depth.

## What We Care About

- How you break down the problem
- Why you made your choices
- Honest assessment of your solution — especially where it fails
- Code someone else can read

## What We Don't Care About

- State-of-the-art performance
- Complex infrastructure
- Perfect solutions
- Exhaustive hyperparameter tuning

We're more interested in your thinking than your metrics.
