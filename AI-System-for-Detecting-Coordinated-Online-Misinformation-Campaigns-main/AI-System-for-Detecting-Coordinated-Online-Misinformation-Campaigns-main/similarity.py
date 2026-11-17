"""Similarity matrix and grouping utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SimilarityPair:
    idx_a: int
    idx_b: int
    score: float


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix."""
    if embeddings.size == 0:
        return np.empty((0, 0))
    return cosine_similarity(embeddings)


def get_similar_pairs(
    sim_matrix: np.ndarray,
    threshold: float = 0.8,
) -> List[SimilarityPair]:
    """Return index pairs where cosine similarity exceeds threshold."""
    pairs: list[SimilarityPair] = []
    n = sim_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if score > threshold:
                pairs.append(SimilarityPair(i, j, score))
    return pairs


def build_similarity_groups(
    similar_pairs: List[SimilarityPair],
    total_items: int,
) -> List[Set[int]]:
    """
    Build connected groups from similar pairs.

    If A is similar to B and B is similar to C, all belong to one group.
    """
    parent = list(range(total_items))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for pair in similar_pairs:
        union(pair.idx_a, pair.idx_b)

    groups_map: dict[int, set[int]] = {}
    for i in range(total_items):
        root = find(i)
        groups_map.setdefault(root, set()).add(i)

    # Keep only groups with > 1 post.
    groups = [members for members in groups_map.values() if len(members) > 1]
    groups.sort(key=len, reverse=True)
    return groups


def build_edges_for_group(
    group: Set[int],
    sim_matrix: np.ndarray,
    threshold: float = 0.8,
) -> List[Tuple[int, int]]:
    """Create index edges in a group where similarity exceeds threshold."""
    members = sorted(group)
    edges: list[tuple[int, int]] = []
    for i, idx_a in enumerate(members):
        for idx_b in members[i + 1 :]:
            if float(sim_matrix[idx_a, idx_b]) > threshold:
                edges.append((idx_a, idx_b))
    return edges
