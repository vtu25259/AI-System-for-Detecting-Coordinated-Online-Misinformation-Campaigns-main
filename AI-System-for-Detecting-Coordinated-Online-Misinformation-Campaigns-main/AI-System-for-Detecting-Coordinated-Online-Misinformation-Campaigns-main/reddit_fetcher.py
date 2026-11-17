"""Simple Reddit fetcher for latest post text and timestamp."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

import pandas as pd
import requests

REDDIT_USER_AGENT = "misinfo-demo/1.0"


def fetch_reddit_posts(subreddit: str = "worldnews", limit: int = 60) -> pd.DataFrame:
    """
    Fetch latest Reddit posts without auth using the public JSON endpoint.

    Returned columns:
        - post_text
        - timestamp
        - source
    """
    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"
    headers = {"User-Agent": REDDIT_USER_AGENT}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    payload = response.json()

    rows: List[dict] = []
    for child in payload.get("data", {}).get("children", []):
        data = child.get("data", {})
        title = (data.get("title") or "").strip()
        selftext = (data.get("selftext") or "").strip()
        text = f"{title} {selftext}".strip()
        if not text:
            continue

        created_utc = data.get("created_utc")
        if created_utc is None:
            continue
        timestamp = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)

        rows.append(
            {
                "post_text": text,
                "timestamp": timestamp,
                "source": "reddit",
            }
        )

    return pd.DataFrame(rows)
