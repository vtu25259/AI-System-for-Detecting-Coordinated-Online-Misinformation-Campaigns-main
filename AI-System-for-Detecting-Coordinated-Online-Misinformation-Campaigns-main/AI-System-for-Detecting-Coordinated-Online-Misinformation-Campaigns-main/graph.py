"""User graph creation and drawing utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def build_user_graph(
    df: pd.DataFrame,
    campaigns: Iterable,
) -> nx.Graph:
    """
    Build graph where nodes are users and edges indicate similar posts.
    """
    graph = nx.Graph()
    edge_weights: dict[tuple[str, str], int] = defaultdict(int)

    for campaign in campaigns:
        for idx_a, idx_b in campaign.edges:
            user_a = str(df.iloc[idx_a]["user_id"])
            user_b = str(df.iloc[idx_b]["user_id"])
            if user_a == user_b:
                continue
            edge_key = tuple(sorted((user_a, user_b)))
            edge_weights[edge_key] += 1

    for (user_a, user_b), weight in edge_weights.items():
        graph.add_edge(user_a, user_b, weight=weight)

    return graph


def draw_user_graph(graph: nx.Graph):
    """Draw graph with cluster colors and return matplotlib figure."""
    fig, ax = plt.subplots(figsize=(10, 7))

    if graph.number_of_nodes() == 0:
        ax.text(
            0.5,
            0.5,
            "No user links available for visualization.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        return fig

    components = list(nx.connected_components(graph))
    palette = [
        "#FF6B6B",
        "#FFD166",
        "#4D96FF",
        "#06D6A0",
        "#B794F4",
        "#F78C6B",
        "#118AB2",
    ]

    node_colors = {}
    for i, component in enumerate(components):
        color = palette[i % len(palette)]
        for node in component:
            node_colors[node] = color

    pos = nx.spring_layout(graph, seed=42, k=0.8)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=[node_colors[n] for n in graph.nodes()],
        node_size=450,
        alpha=0.9,
        ax=ax,
    )
    nx.draw_networkx_edges(graph, pos, alpha=0.35, width=1.2, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)

    ax.set_title("User Similarity Network (Cluster Highlighting)")
    ax.axis("off")
    plt.tight_layout()
    return fig
