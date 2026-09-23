"""FAISS vector indexing and multi-query context retrieval."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple
import faiss
import numpy as np

from analyst.embeddings import embed_passages, embed_queries
from analyst.models import Exchange

logger = logging.getLogger("analyst.index")


class IndexStore:
    """In-memory multi-transcript vector store using one FAISS IndexFlatIP per transcript."""

    def __init__(self) -> None:
        self._indices: Dict[str, faiss.IndexFlatIP] = {}
        self._exchanges: Dict[str, List[Exchange]] = {}
        self._chunk_map: Dict[str, Exchange] = {}

    def add_transcript(self, exchanges: List[Exchange]) -> None:
        """Embed and index exchanges for a single transcript."""
        if not exchanges:
            return

        tid = exchanges[0].transcript_id
        texts = [ex.embed_text for ex in exchanges]
        vectors = embed_passages(texts)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        self._indices[tid] = index
        self._exchanges[tid] = list(exchanges)

        for ex in exchanges:
            self._chunk_map[ex.chunk_id] = ex

    def transcript_ids(self) -> List[str]:
        """Return list of indexed transcript IDs sorted numerically."""
        return sorted(
            list(self._indices.keys()),
            key=lambda tid: int(tid[1:]) if tid.startswith("T") and tid[1:].isdigit() else tid,
        )

    def exchanges(self, tid: str) -> List[Exchange]:
        """Return all exchanges belonging to a transcript ID in sequential order."""
        if tid not in self._exchanges:
            raise KeyError(f"Transcript ID '{tid}' is not registered in IndexStore.")
        return list(self._exchanges[tid])

    def get(self, chunk_id: str) -> Exchange:
        """Lookup an exchange chunk by its chunk_id (e.g. 'T1-X3')."""
        if chunk_id not in self._chunk_map:
            raise KeyError(f"Exchange chunk_id '{chunk_id}' not found.")
        return self._chunk_map[chunk_id]

    def search(self, tid: str, query_vecs: np.ndarray, k: int) -> List[Tuple[Exchange, float]]:
        """Search a single transcript's index using pre-computed query vectors.

        Args:
            tid: Transcript identifier (e.g., 'T1').
            query_vecs: 2D float32 array of shape (num_queries, dim).
            k: Number of nearest neighbors to retrieve per query.

        Returns:
            Flat list of (Exchange, score) tuples deduplicated by best score.
        """
        if tid not in self._indices:
            raise KeyError(f"Transcript ID '{tid}' is not registered in IndexStore.")

        if query_vecs.shape[0] == 0:
            return []

        index = self._indices[tid]
        total_items = index.ntotal
        actual_k = min(k, total_items)
        if actual_k <= 0:
            return []

        scores, indices = index.search(query_vecs, actual_k)

        best_hits: Dict[str, Tuple[Exchange, float]] = {}
        for q_idx in range(scores.shape[0]):
            for rank in range(actual_k):
                idx = int(indices[q_idx, rank])
                if idx < 0:
                    continue
                score = float(scores[q_idx, rank])
                ex = self._exchanges[tid][idx]
                if ex.chunk_id not in best_hits or score > best_hits[ex.chunk_id][1]:
                    best_hits[ex.chunk_id] = (ex, score)

        # Sort by score descending
        return sorted(best_hits.values(), key=lambda item: item[1], reverse=True)


def retrieve(
    store: IndexStore,
    queries: List[str],
    transcript_ids: List[str],
    k_per_query: int = 3,
    neighbours: int = 1,
    max_chunks: int = 8,
) -> Dict[str, List[Exchange]]:
    """Retrieve relevant exchanges across transcripts with context window expansion.

    For each transcript:
      1. Embed all queries at once.
      2. Take top k_per_query hits per query, keeping their best scores.
      3. Expand each hit by +/- `neighbours` adjacent exchanges.
      4. Deduplicate.
      5. Cap at max_chunks (prioritizing direct hits by score, then neighbours).
      6. Return results in native transcript exchange order.

    Raises:
        KeyError: If any requested transcript ID is not present in store.
    """
    for tid in transcript_ids:
        # Validate existence using public method (raises KeyError if missing)
        store.exchanges(tid)

    if not queries or not transcript_ids:
        return {tid: [] for tid in transcript_ids}

    query_vecs = embed_queries(queries)
    results: Dict[str, List[Exchange]] = {}

    for tid in transcript_ids:
        tid_exchanges = store.exchanges(tid)
        n_total = len(tid_exchanges)

        # Search index
        hits = store.search(tid, query_vecs, k=k_per_query)

        # Map each exchange chunk_id directly to its index in the transcript
        id_to_idx = {ex.chunk_id: i for i, ex in enumerate(tid_exchanges)}

        # Build set of candidate indices (including neighbours)
        candidate_indices: set[int] = set()
        neighbour_scores: Dict[int, float] = {}

        for ex, score in hits:
            ex_idx = id_to_idx[ex.chunk_id]
            candidate_indices.add(ex_idx)

            # Add +/- neighbours
            for offset in range(-neighbours, neighbours + 1):
                n_idx = ex_idx + offset
                if 0 <= n_idx < n_total:
                    candidate_indices.add(n_idx)
                    # Score neighbours slightly lower than the seed hit to prioritize direct hits on capping
                    adj_score = score - (abs(offset) * 0.001)
                    if n_idx not in neighbour_scores or adj_score > neighbour_scores[n_idx]:
                        neighbour_scores[n_idx] = adj_score

        # If total candidates exceed max_chunks, keep top candidates by score
        if len(candidate_indices) > max_chunks:
            sorted_by_score = sorted(
                candidate_indices,
                key=lambda idx: neighbour_scores.get(idx, -1.0),
                reverse=True,
            )
            selected_indices = set(sorted_by_score[:max_chunks])
        else:
            selected_indices = candidate_indices

        # Sort selected exchanges chronologically by transcript order
        chronological_indices = sorted(list(selected_indices))
        results[tid] = [tid_exchanges[idx] for idx in chronological_indices]

    return results