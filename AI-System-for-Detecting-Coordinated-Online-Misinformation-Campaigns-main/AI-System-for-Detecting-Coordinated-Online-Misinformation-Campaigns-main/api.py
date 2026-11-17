"""FastAPI backend for coordinated misinformation detection."""

from __future__ import annotations

from io import StringIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from data_loader import validate_posts_df
from detection import detect_campaigns
from embedding import embed_texts, load_embedding_model
from similarity import build_similarity_groups, compute_similarity_matrix, get_similar_pairs

app = FastAPI(title="Coordinated Narrative Detection API", version="1.0.0")
model = load_embedding_model()


def run_detection_pipeline(df: pd.DataFrame) -> dict:
    """Run full detection pipeline and return serializable results."""
    validated = validate_posts_df(df)
    embeddings = embed_texts(model, validated["post_text"].tolist())
    sim_matrix = compute_similarity_matrix(embeddings)
    similar_pairs = get_similar_pairs(sim_matrix, threshold=0.8)
    groups = build_similarity_groups(similar_pairs, total_items=len(validated))
    campaigns = detect_campaigns(
        validated,
        groups,
        sim_matrix=sim_matrix,
        similarity_threshold=0.8,
        max_window_minutes=10,
        min_users=5,
    )

    return {
        "summary": {
            "total_posts": int(len(validated)),
            "total_users": int(validated["user_id"].nunique()),
            "campaigns_detected": len(campaigns),
        },
        "campaigns": [campaign.to_dict() for campaign in campaigns],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        content = (await file.read()).decode("utf-8")
        df = pd.read_csv(StringIO(content))
        result = run_detection_pipeline(df)
        return result
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail=str(exc)) from exc
