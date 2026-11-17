"""Campaign detection rules and human-readable messaging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from similarity import build_edges_for_group


@dataclass
class CampaignResult:
    risk_level: str
    alert_message: str
    users_count: int
    posts_count: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_minutes: float
    example_text: str
    group_indices: list[int]
    edges: list[tuple[int, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "alert_message": self.alert_message,
            "users_count": self.users_count,
            "posts_count": self.posts_count,
            "start_time": str(self.start_time),
            "end_time": str(self.end_time),
            "duration_minutes": round(self.duration_minutes, 2),
            "example_text": self.example_text,
            "group_indices": self.group_indices,
            "edges": self.edges,
        }


def classify_risk(unique_users: int) -> str:
    """Classify campaign risk based on unique users in cluster."""
    if unique_users >= 20:
        return "HIGH"
    if 10 <= unique_users <= 19:
        return "MEDIUM"
    return "LOW"


def generate_alert_message(risk_level: str, users_count: int, minutes: float) -> str:
    """Generate presentation-friendly campaign alert text."""
    rounded_minutes = max(1, int(round(minutes)))
    if risk_level == "HIGH":
        return (
            f'🚨 "{users_count} accounts posted highly similar content within '
            f'{rounded_minutes} minutes -> STRONG COORDINATION SIGNAL"'
        )
    if risk_level == "MEDIUM":
        return (
            f'⚠️ "{users_count} accounts showing coordinated behavior within '
            'a short time window"'
        )
    return (
        f'ℹ️ "Small cluster detected: {users_count} users posting similar content"'
    )


def detect_campaigns(
    df: pd.DataFrame,
    groups: list[set[int]],
    sim_matrix,
    similarity_threshold: float = 0.8,
    max_window_minutes: int = 10,
    min_users: int = 5,
) -> list[CampaignResult]:
    """Filter similarity groups using time and user thresholds."""
    campaigns: list[CampaignResult] = []

    for group in groups:
        group_indices = sorted(group)
        subset = df.iloc[group_indices].copy()

        start_time = subset["timestamp"].min()
        end_time = subset["timestamp"].max()
        duration_minutes = (end_time - start_time).total_seconds() / 60.0

        unique_users = subset["user_id"].nunique()
        posts_count = len(subset)

        if duration_minutes > max_window_minutes:
            continue
        if unique_users < min_users:
            continue

        risk_level = classify_risk(unique_users)
        alert_message = generate_alert_message(risk_level, unique_users, duration_minutes)
        sample_text = str(subset["post_text"].iloc[0])[:180]
        edges = build_edges_for_group(group, sim_matrix, threshold=similarity_threshold)

        campaigns.append(
            CampaignResult(
                risk_level=risk_level,
                alert_message=alert_message,
                users_count=unique_users,
                posts_count=posts_count,
                start_time=start_time,
                end_time=end_time,
                duration_minutes=duration_minutes,
                example_text=sample_text,
                group_indices=group_indices,
                edges=edges,
            )
        )

    campaigns.sort(key=lambda c: c.users_count, reverse=True)
    return campaigns
