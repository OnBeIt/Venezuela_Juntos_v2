"""Thin wrapper around facerec that works against the SQLite DB."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "lib"))
import facerec  # noqa: E402

EMBEDDING_DIM = 512
THRESHOLD = facerec.DEFAULT_MATCH_THRESHOLD


def embed(image_bytes: bytes) -> np.ndarray:
    """Return the 512-d normed embedding of the largest face in the image.

    Raises ValueError if no face is detected.
    """
    records = facerec.analyze_bytes(image_bytes)
    if not records:
        raise ValueError("No face detected in the uploaded image.")
    return records[0].normed_embedding.astype(np.float32)


def embedding_to_blob(emb: np.ndarray) -> bytes:
    return emb.astype(np.float32).tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def search_pool(probe: np.ndarray, pool: list[tuple]) -> list[tuple]:
    """Find matches above threshold in a pool of (record, embedding_blob) pairs.

    Returns list of (record, similarity) sorted descending, threshold-filtered.
    """
    if not pool:
        return []

    matrix = np.vstack([blob_to_embedding(blob) for _, blob in pool])  # (N, 512)
    sims = matrix @ probe  # cosine similarity (both L2-normed)

    results = []
    for (record, _), sim in zip(pool, sims):
        if float(sim) >= THRESHOLD:
            results.append((record, round(float(sim), 4)))

    results.sort(key=lambda t: t[1], reverse=True)
    return results
