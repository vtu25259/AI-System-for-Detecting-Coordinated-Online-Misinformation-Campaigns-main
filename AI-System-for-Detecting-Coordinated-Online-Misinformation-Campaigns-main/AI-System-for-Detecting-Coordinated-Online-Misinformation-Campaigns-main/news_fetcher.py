"""Simple News data fetcher with environment-key support and safe fallback."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests


def fetch_news_posts(limit: int = 40) -> pd.DataFrame:
    """
    Fetch latest news headlines.

    If NEWS_API_KEY is not available or request fails, return small mock news data.
    """
    api_key = os.getenv("361d9d0f491849e3a7a1e622ba14e9b9")
    if not api_key:
        return _mock_news(limit=limit)

    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "country": "us",
        "language": "en",
        "pageSize": min(limit, 100),
        "apiKey": api_key,
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        rows = []
        for article in payload.get("articles", []):
            title = (article.get("title") or "").strip()
            desc = (article.get("description") or "").strip()
            text = f"{title} {desc}".strip()
            if not text:
                continue
            published_at = article.get("publishedAt")
            if not published_at:
                continue
            timestamp = pd.to_datetime(published_at, errors="coerce", utc=True)
            if pd.isna(timestamp):
                continue
            source_name = article.get("source", {}).get("name", "news")
            rows.append(
                {
                    "post_text": text,
                    "timestamp": timestamp,
                    "source": source_name or "news",
                }
            )
        if not rows:
            return _mock_news(limit=limit)
        return pd.DataFrame(rows)
    except Exception:
        return _mock_news(limit=limit)


def _mock_news(limit: int = 40) -> pd.DataFrame:
    """Build mock news-like posts for reliable demo mode."""
    now = datetime.now(timezone.utc)
    base_texts = [
        "Breaking crypto market alert as volatility rises sharply",
        "Urgent financial warning issued after sudden market movement",
        "Bitcoin will crash soon says market risk analyst",
        "Global markets react to coordinated narrative spread online",
    ]
    rows = []
    for idx in range(max(12, min(limit, 40))):
        text = base_texts[idx % len(base_texts)]
        rows.append(
            {
                "post_text": text,
                "timestamp": now - timedelta(minutes=(idx % 9)),
                "source": "news",
            }
        )
    return pd.DataFrame(rows)
