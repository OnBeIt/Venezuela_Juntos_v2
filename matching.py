"""Thin wrapper around facerec that works against the SQLite DB."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "lib"))
import facerec  # noqa: E402

EMBEDDING_DIM = 512
THRESHOLD = facerec.DEFAULT_MATCH_THRESHOLD

# Quality heuristics for the largest detected face. A photo is flagged (warn,
# don't block) when any of these fail — flagged photos still get embedded and
# stored, but the user is warned that match quality may suffer. Tunable.
MIN_BLUR_VAR = 40.0      # variance of Laplacian; lower = blurrier
MIN_FACE_PX = 80 * 80    # detected face area in pixels; smaller = too distant
MIN_BRIGHTNESS = 50.0    # mean luma 0..255; lower = too dark
MAX_BRIGHTNESS = 205.0   # higher = blown out / overexposed


def warmup() -> None:
    """Load the InsightFace model into memory. Call once at startup."""
    facerec.get_app()


def embed(image_bytes: bytes) -> np.ndarray:
    """Return the 512-d normed embedding of the largest face in the image.

    Raises ValueError if no face is detected.
    """
    records = facerec.analyze_bytes(image_bytes)
    if not records:
        raise ValueError("No face detected in the uploaded image.")
    return records[0].normed_embedding.astype(np.float32)


def _assess_quality(record) -> tuple[bool, str | None]:
    """Return (low_quality, note) for the largest face's quality estimates."""
    q = record.quality
    reasons = []
    if q.get("blur_var", 0.0) < MIN_BLUR_VAR:
        reasons.append("blurry")
    if q.get("face_px", 0.0) < MIN_FACE_PX:
        reasons.append("face too small")
    brightness = q.get("brightness", 128.0)
    if brightness < MIN_BRIGHTNESS:
        reasons.append("too dark")
    elif brightness > MAX_BRIGHTNESS:
        reasons.append("overexposed")
    if not reasons:
        return False, None
    return True, ", ".join(reasons)


def analyze(image_bytes: bytes) -> tuple[np.ndarray, bool, str | None]:
    """Return (normed embedding, low_quality, quality_note) for the largest face.

    Raises ValueError if no face is detected.
    """
    records = facerec.analyze_bytes(image_bytes)
    if not records:
        raise ValueError("No face detected in the uploaded image.")
    primary = records[0]
    low_quality, note = _assess_quality(primary)
    return primary.normed_embedding.astype(np.float32), low_quality, note


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
