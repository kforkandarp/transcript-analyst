"""Tests for FAISS IndexStore and multi-query retrieval on real transcript data."""

from pathlib import Path
import pytest

from analyst.chunker import build_exchanges
from analyst.embeddings import embed_queries
from analyst.index import IndexStore, retrieve
from analyst.parser import load_raw_dir

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.fixture(scope="module")
def populated_store() -> IndexStore:
    """Parse real transcripts and populate an IndexStore instance."""
    transcripts, _ = load_raw_dir(DATA_RAW_DIR)
    store = IndexStore()
    for t in transcripts:
        exchanges = build_exchanges(t)
        store.add_transcript(exchanges)
    return store


def test_search_assertions(populated_store: IndexStore) -> None:
    """Verify semantic search ranking across individual transcripts."""
    # 1. Query "How long does a purchase decision take?" over T1 has T1-X7 in top 2 hits
    q1_vec = embed_queries(["How long does a purchase decision take?"])
    hits_t1 = populated_store.search("T1", q1_vec, k=2)
    hit_ids_t1 = [ex.chunk_id for ex, _ in hits_t1]
    assert "T1-X7" in hit_ids_t1

    # 2. Query "surgeon training" over T2 has T2-X4 in top 2 hits
    q2_vec = embed_queries(["surgeon training"])
    hits_t2 = populated_store.search("T2", q2_vec, k=2)
    hit_ids_t2 = [ex.chunk_id for ex, _ in hits_t2]
    assert "T2-X4" in hit_ids_t2

    # 3. Query "expected growth over the next years" over T3 has T3-X5 in top 3 hits
    q3_vec = embed_queries(["expected growth over the next years"])
    hits_t3 = populated_store.search("T3", q3_vec, k=3)
    hit_ids_t3 = [ex.chunk_id for ex, _ in hits_t3]
    assert "T3-X5" in hit_ids_t3


def test_retrieve_neighbours_and_deduplication(populated_store: IndexStore) -> None:
    """Verify retrieve() returns neighbouring exchanges without duplicates in transcript order."""
    results = retrieve(
        store=populated_store,
        queries=["How long does a purchase decision take?"],
        transcript_ids=["T1"],
        k_per_query=1,
        neighbours=1,
        max_chunks=8,
    )

    t1_chunks = results["T1"]
    chunk_ids = [ex.chunk_id for ex in t1_chunks]

    # Must contain no duplicates
    assert len(chunk_ids) == len(set(chunk_ids))

    # T1-X7 is the direct hit, so neighbours=1 must include T1-X6
    assert "T1-X7" in chunk_ids
    assert "T1-X6" in chunk_ids

    # Must be sorted in chronological order (T1-X6 before T1-X7)
    indices = [int(cid.split("-X")[1]) for cid in chunk_ids]
    assert indices == sorted(indices)


def test_multi_transcript_retrieve_and_error_handling(populated_store: IndexStore) -> None:
    """Verify multi-transcript queries and unknown ID handling."""
    # Two-transcript retrieve returns results for both IDs
    res = retrieve(
        store=populated_store,
        queries=["robotics adoption"],
        transcript_ids=["T1", "T2"],
        k_per_query=2,
    )
    assert "T1" in res and len(res["T1"]) > 0
    assert "T2" in res and len(res["T2"]) > 0

    # Unknown ID raises KeyError with informative message
    with pytest.raises(KeyError) as exc_info:
        retrieve(populated_store, queries=["test"], transcript_ids=["T99"])
    assert "T99" in str(exc_info.value)