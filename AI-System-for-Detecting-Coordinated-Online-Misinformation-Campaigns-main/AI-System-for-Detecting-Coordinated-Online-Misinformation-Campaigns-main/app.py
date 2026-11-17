"""Single-post demo mode for coordinated narrative detection."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from data_merger import merge_sources
from embedding import embed_texts, load_embedding_model
from news_fetcher import fetch_news_posts
from reddit_fetcher import fetch_reddit_posts

SIMILARITY_THRESHOLD = 0.6
TIME_WINDOW_MINUTES = 10

st.set_page_config(page_title="Coordinated Narrative Detection", page_icon="🚨", layout="wide")


@dataclass
class AnalysisResult:
    similar_count: int
    start_time: pd.Timestamp | None
    end_time: pd.Timestamp | None
    avg_similarity: float
    message: str
    similar_df: pd.DataFrame
    graph: nx.Graph


@st.cache_resource
def get_model():
    return load_embedding_model()


def fetch_live_data(subreddit: str) -> pd.DataFrame:
    """Fetch Reddit + optional News data and normalize shape."""
    frames = []
    try:
        frames.append(fetch_reddit_posts(subreddit=subreddit, limit=80))
    except Exception:
        frames.append(pd.DataFrame(columns=["post_text", "timestamp", "source"]))

    # News fetch auto-falls back to mock data when API key or request is unavailable.
    frames.append(fetch_news_posts(limit=30))
    merged = merge_sources(frames)
    return merged.head(120).copy()


def build_message(similar_count: int, within_window: bool) -> str:
    if similar_count >= 12 and within_window:
        return (
            "🚨 Multiple posts similar to your input appeared within minutes "
            "-> POSSIBLE COORDINATED NARRATIVE"
        )
    if similar_count >= 5 and within_window:
        return "⚠️ Several similar posts detected in a short time window"
    return "ℹ️ Few similar posts found — low coordination signal"


def analyze_input_post(input_text: str, fetched_df: pd.DataFrame) -> AnalysisResult:
    model = get_model()
    corpus_texts = fetched_df["post_text"].tolist()
    all_texts = [input_text] + corpus_texts
    embeddings = embed_texts(model, all_texts)

    input_embedding = embeddings[0:1]
    corpus_embeddings = embeddings[1:]
    sim_scores = cosine_similarity(input_embedding, corpus_embeddings)[0]

    scored_df = fetched_df.copy()
    scored_df["similarity"] = sim_scores
    similar_df = scored_df[scored_df["similarity"] > SIMILARITY_THRESHOLD].copy()
    similar_df = similar_df.sort_values("similarity", ascending=False).reset_index(drop=True)

    if similar_df.empty:
        graph = nx.Graph()
        graph.add_node("INPUT_POST")
        return AnalysisResult(
            similar_count=0,
            start_time=None,
            end_time=None,
            avg_similarity=0.0,
            message=build_message(0, within_window=False),
            similar_df=similar_df,
            graph=graph,
        )

    start_time = similar_df["timestamp"].min()
    end_time = similar_df["timestamp"].max()
    duration_minutes = (end_time - start_time).total_seconds() / 60.0
    within_window = duration_minutes <= TIME_WINDOW_MINUTES
    avg_similarity = float(similar_df["similarity"].mean())
    message = build_message(len(similar_df), within_window=within_window)
    graph = build_post_graph(input_text, similar_df)

    return AnalysisResult(
        similar_count=len(similar_df),
        start_time=start_time,
        end_time=end_time,
        avg_similarity=avg_similarity,
        message=message,
        similar_df=similar_df,
        graph=graph,
    )


def build_post_graph(input_text: str, similar_df: pd.DataFrame) -> nx.Graph:
    """Create post-level graph: input node connected to similar nodes."""
    graph = nx.Graph()
    graph.add_node("INPUT_POST", label=input_text[:45] + ("..." if len(input_text) > 45 else ""))

    texts = similar_df["post_text"].tolist()
    if not texts:
        return graph

    for idx, row in similar_df.iterrows():
        node_id = f"POST_{idx}"
        label = row["post_text"][:45] + ("..." if len(row["post_text"]) > 45 else "")
        graph.add_node(node_id, label=label, similarity=float(row["similarity"]))
        graph.add_edge("INPUT_POST", node_id, weight=float(row["similarity"]))

    sim_embeddings = embed_texts(get_model(), texts)
    matrix = cosine_similarity(sim_embeddings)
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            if float(matrix[i, j]) > SIMILARITY_THRESHOLD:
                graph.add_edge(f"POST_{i}", f"POST_{j}", weight=float(matrix[i, j]))

    return graph


def draw_graph(graph: nx.Graph):
    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(graph, seed=42)

    input_nodes = [n for n in graph.nodes if n == "INPUT_POST"]
    similar_nodes = [n for n in graph.nodes if n != "INPUT_POST"]

    nx.draw_networkx_nodes(graph, pos, nodelist=input_nodes, node_color="#EF4444", node_size=900, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=similar_nodes, node_color="#3B82F6", node_size=550, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.4, width=1.4, ax=ax)

    labels = {n: ("INPUT" if n == "INPUT_POST" else n.replace("POST_", "P")) for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)

    ax.set_title("Similarity Graph (Input Post vs Similar Recent Posts)")
    ax.axis("off")
    plt.tight_layout()
    return fig


def render_message_card(message: str):
    if message.startswith("🚨"):
        bg, border = "#FEE2E2", "#B91C1C"
    elif message.startswith("⚠️"):
        bg, border = "#FEF9C3", "#92400E"
    else:
        bg, border = "#DBEAFE", "#1D4ED8"

    st.markdown(
        f"""
        <div style="
            background-color: {bg};
            border-left: 6px solid {border};
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 12px;">
            <div style="font-size: 1.05rem; font-weight: 700;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.title("Coordinated Narrative Detection System")
    st.markdown("## 🚨 Simple Demo Mode")
    st.caption("Enter one post and detect similar narratives from recent live data.")

    subreddit = st.selectbox("Reddit source", ["worldnews", "cryptocurrency", "politics"], index=0)
    input_text = st.text_area("Enter a post to analyze", value="Bitcoin will crash soon", height=120)
    run = st.button("Analyze Post", type="primary")

    if not run:
        st.stop()
    if not input_text.strip():
        st.warning("Please enter text to analyze.")
        st.stop()

    with st.spinner("Fetching recent posts and running similarity analysis..."):
        live_df = fetch_live_data(subreddit=subreddit)
        if live_df.empty:
            st.error("Could not fetch any posts. Please try again.")
            st.stop()
        result = analyze_input_post(input_text.strip(), live_df)

    render_message_card(result.message)

    col1, col2, col3 = st.columns(3)
    col1.metric("Similar Posts Found", result.similar_count)
    if result.start_time is not None and result.end_time is not None:
        minutes = max(0.0, (result.end_time - result.start_time).total_seconds() / 60.0)
        col2.metric("Time Range (minutes)", f"{minutes:.1f}")
    else:
        col2.metric("Time Range (minutes)", "N/A")
    col3.metric("Avg Similarity", f"{result.avg_similarity:.2f}")

    st.markdown("### Similar Posts by Source")
    if result.similar_df.empty:
        st.info("No coordinated narrative detected across sources")
    else:
        similar_source_counts = result.similar_df["source"].value_counts().rename("similar_posts")
        st.bar_chart(similar_source_counts)

    st.markdown("### Similarity Graph")
    st.pyplot(draw_graph(result.graph), use_container_width=True)

    st.markdown("### Sample Similar Posts")
    preview_cols = ["post_text", "timestamp", "source", "similarity"]
    if result.similar_df.empty:
        st.info("No posts crossed the similarity threshold (> 0.7).")
    else:
        st.dataframe(result.similar_df[preview_cols].head(15), use_container_width=True)


if __name__ == "__main__":
    main()
