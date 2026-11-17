# AI System for Detecting Coordinated Misinformation Campaigns

A simple, presentation-ready Streamlit demo:

1. User enters one post.
2. System fetches recent posts (Reddit + lightweight News fallback).
3. It finds similar posts using sentence embeddings.
4. It checks time concentration (10-minute window) and shows a coordination signal.

## Demo Flow

- Input: `Enter a post to analyze`
- Action: `Analyze Post`
- Output:
  - main alert message (high / medium / low signal)
  - number of similar posts
  - time range
  - average similarity score
  - sample similar texts
  - NetworkX graph (input post highlighted)

## Detection Logic

- Embedding model: `all-MiniLM-L6-v2`
- Similarity threshold: `> 0.7`
- Temporal window: `<= 10 minutes`
- Message rules:
  - High: many similar posts in short time window
  - Medium: several similar posts in short time window
  - Low: few similar posts

## Data Sources

- Reddit live feed (`/r/worldnews`, `/r/cryptocurrency`, `/r/politics`)
- News headlines:
  - uses `NEWS_API_KEY` if available
  - otherwise falls back to mock headlines for reliable demos

## Files

- `app.py` - Streamlit single-post demo
- `reddit_fetcher.py` - latest Reddit posts
- `news_fetcher.py` - NewsAPI + fallback mock
- `data_merger.py` - normalize and merge sources
- `embedding.py` - sentence-transformer utilities
- `requirements.txt` - dependencies

## Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Notes

- This is a demo system, not production-grade moderation.
- Internet/API availability may vary; fallback data keeps the app usable for presentations.
