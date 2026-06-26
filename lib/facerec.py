#!/usr/bin/env python3
"""
facerec — facial recognition toolkit built on InsightFace (buffalo_l / ArcFace).

Capabilities
------------
  analyze   Extract biometric data from an image: bounding box, detection
            confidence, 2D/3D landmarks, derived facial geometry, estimated
            age/gender/pose, an image-quality estimate, and the 512-d
            recognition embedding.
  verify    1:1 — compare two images and report whether they are the same person.
  enroll    Add one or more images of a named person to a local gallery.
  search    1:N — find the closest enrolled person(s) for a query image.
  gallery   List / remove entries in the gallery.

The recognition embedding is an L2-normalized 512-d ArcFace vector; faces are
compared with cosine similarity (equivalently, dot product of normed vectors).

Usage examples
--------------
  python facerec.py analyze photo.jpg
  python facerec.py analyze photo.jpg --full --save-crops out/
  python facerec.py verify a.jpg b.jpg
  python facerec.py enroll "Ada Lovelace" img1.jpg img2.jpg
  python facerec.py search query.jpg --top 5
  python facerec.py gallery --list

Note: face attributes (age/gender/pose) are statistical *estimates*, not facts.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODEL_NAME = "buffalo_l"          # InsightFace model pack (detection + landmarks + genderage + recognition)
DET_SIZE = (640, 640)             # detector input size; larger = slower but finds smaller faces
EMBEDDING_DIM = 512

# Cosine-similarity decision threshold for the buffalo_l (R50, glint360k) model.
# Above this two faces are treated as the same identity. ~0.40 is a balanced
# default; raise it to reduce false matches, lower it to reduce missed matches.
DEFAULT_MATCH_THRESHOLD = 0.40

GALLERY_PATH = Path(os.environ.get("FACEREC_GALLERY", "gallery.json"))

_APP = None  # lazily-initialized FaceAnalysis singleton


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def get_app():
    """Return a ready FaceAnalysis app, loading the model pack on first use.

    The model pack (~280MB) is downloaded once to ~/.insightface/models and
    cached. InsightFace is chatty on stdout while loading; we suppress that so
    it does not pollute JSON output.
    """
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis

        with contextlib.redirect_stdout(io.StringIO()):
            app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=DET_SIZE)
        _APP = app
    return _APP


def _imread(path: str | Path) -> np.ndarray:
    """Read an image as a BGR numpy array, raising a clear error on failure."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def expand_images(paths: list[str | Path]) -> list[Path]:
    """Expand a list of files and/or directories into a sorted list of image files.

    Directories are scanned (non-recursively) for supported image extensions.
    """
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(q for q in p.iterdir()
                              if q.suffix.lower() in IMAGE_EXTS))
        else:
            out.append(p)
    return out


# --------------------------------------------------------------------------- #
# Biometric extraction
# --------------------------------------------------------------------------- #

@dataclass
class FaceRecord:
    """Structured biometric data for a single detected face."""

    index: int
    bbox: list[float]                      # [x1, y1, x2, y2]
    det_score: float                       # detection confidence 0..1
    age: int | None
    gender: str | None                     # "M" / "F"
    pose: dict[str, float] | None          # yaw / pitch / roll in degrees
    geometry: dict[str, float]             # derived measurements (pixels / ratios)
    quality: dict[str, float]              # quality estimates
    kps5: list[list[float]]                # 5 keypoints: L-eye, R-eye, nose, L-mouth, R-mouth
    landmarks_2d_106: list[list[float]] | None
    landmarks_3d_68: list[list[float]] | None
    embedding: np.ndarray = field(repr=False)        # raw 512-d
    normed_embedding: np.ndarray = field(repr=False)  # L2-normalized 512-d

    def to_dict(self, include_embedding: bool = False,
                include_dense_landmarks: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "bbox": [round(v, 2) for v in self.bbox],
            "det_score": round(self.det_score, 4),
            "age": self.age,
            "gender": self.gender,
            "pose": {k: round(v, 2) for k, v in self.pose.items()} if self.pose else None,
            "geometry": {k: round(v, 4) for k, v in self.geometry.items()},
            "quality": {k: round(v, 4) for k, v in self.quality.items()},
            "keypoints_5": [[round(x, 2) for x in p] for p in self.kps5],
        }
        if include_dense_landmarks:
            d["landmarks_2d_106"] = self.landmarks_2d_106
            d["landmarks_3d_68"] = self.landmarks_3d_68
        if include_embedding:
            d["embedding"] = [round(float(x), 6) for x in self.embedding]
        else:
            d["embedding_dim"] = int(self.embedding.shape[0])
        return d


def _derive_geometry(bbox: np.ndarray, kps5: np.ndarray) -> dict[str, float]:
    """Compute simple facial measurements from the bbox and 5 keypoints.

    kps5 order (InsightFace): left eye, right eye, nose, left mouth, right mouth.
    Ratios are normalized by interocular distance so they are scale-invariant.
    """
    left_eye, right_eye, nose, left_mouth, right_mouth = kps5
    interocular = float(np.linalg.norm(right_eye - left_eye))
    eyes_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (left_mouth + right_mouth) / 2.0

    face_w = float(bbox[2] - bbox[0])
    face_h = float(bbox[3] - bbox[1])

    geom = {
        "interocular_px": interocular,
        "face_width_px": face_w,
        "face_height_px": face_h,
        "face_aspect_ratio": (face_w / face_h) if face_h else 0.0,
        "eye_to_mouth_px": float(np.linalg.norm(eyes_mid - mouth_mid)),
        "nose_to_eyes_px": float(np.linalg.norm(nose - eyes_mid)),
        "mouth_width_px": float(np.linalg.norm(right_mouth - left_mouth)),
    }
    if interocular:
        geom["eye_to_mouth_ratio"] = geom["eye_to_mouth_px"] / interocular
        geom["mouth_width_ratio"] = geom["mouth_width_px"] / interocular
        geom["face_width_ratio"] = face_w / interocular
    return geom


def _estimate_quality(img: np.ndarray, bbox: np.ndarray, det_score: float) -> dict[str, float]:
    """Cheap quality heuristics for the detected face crop."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(max(0, bbox[0])), int(max(0, bbox[1])),
                      int(min(w, bbox[2])), int(min(h, bbox[3])))
    crop = img[y1:y2, x1:x2]
    quality = {
        "det_score": float(det_score),
        "face_px": float((x2 - x1) * (y2 - y1)),
        "blur_var": 0.0,        # variance of Laplacian; higher = sharper
        "brightness": 0.0,      # mean luma 0..255
    }
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        quality["blur_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        quality["brightness"] = float(gray.mean())
    return quality


def _pose_from_face(face) -> dict[str, float] | None:
    """Extract yaw/pitch/roll if the model estimated head pose."""
    pose = getattr(face, "pose", None)
    if pose is None:
        return None
    pitch, yaw, roll = (float(v) for v in pose)
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


def _gender_str(face) -> str | None:
    sex = getattr(face, "sex", None)
    if sex in ("M", "F"):
        return sex
    g = getattr(face, "gender", None)
    if g is None:
        return None
    return "M" if int(g) == 1 else "F"


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes (e.g. from a download) into a BGR numpy array."""
    if hasattr(data, "read"):          # accept a file-like / BytesIO too
        data = data.read()
    if hasattr(data, "getvalue"):      # BytesIO that wasn't read
        data = data.getvalue()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes (unsupported or corrupt format)")
    return img


def analyze_image(path: str | Path) -> list[FaceRecord]:
    """Detect all faces in an image file and extract full biometric data for each."""
    return analyze_array(_imread(path))


def analyze_bytes(data: bytes) -> list[FaceRecord]:
    """Same as analyze_image but for in-memory image bytes (no file on disk)."""
    return analyze_array(decode_image(data))


def analyze_array(img: np.ndarray) -> list[FaceRecord]:
    """Detect all faces in a BGR image array and extract full biometric data."""
    app = get_app()
    faces = app.get(img)
    # Sort largest-face-first so index 0 is the most prominent subject.
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                   reverse=True)

    records: list[FaceRecord] = []
    for i, f in enumerate(faces):
        kps5 = np.asarray(f.kps, dtype=float)
        emb = np.asarray(f.embedding, dtype=float)
        normed = np.asarray(getattr(f, "normed_embedding", emb / (np.linalg.norm(emb) + 1e-12)),
                            dtype=float)
        lm106 = getattr(f, "landmark_2d_106", None)
        lm68 = getattr(f, "landmark_3d_68", None)
        records.append(FaceRecord(
            index=i,
            bbox=[float(v) for v in f.bbox],
            det_score=float(f.det_score),
            age=int(f.age) if getattr(f, "age", None) is not None else None,
            gender=_gender_str(f),
            pose=_pose_from_face(f),
            geometry=_derive_geometry(np.asarray(f.bbox, dtype=float), kps5),
            quality=_estimate_quality(img, np.asarray(f.bbox, dtype=float), float(f.det_score)),
            kps5=kps5.tolist(),
            landmarks_2d_106=lm106.tolist() if lm106 is not None else None,
            landmarks_3d_68=lm68.tolist() if lm68 is not None else None,
            embedding=emb,
            normed_embedding=normed,
        ))
    return records


def primary_embedding(path: str | Path) -> np.ndarray:
    """Return the normed embedding of the most prominent face, or raise if none."""
    records = analyze_image(path)
    if not records:
        raise ValueError(f"No face detected in {path}")
    return records[0].normed_embedding


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1] between two embeddings."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


# --------------------------------------------------------------------------- #
# Gallery (1:N enrollment store)
# --------------------------------------------------------------------------- #

class Gallery:
    """A simple JSON-backed store of enrolled identities and their embeddings.

    Schema: {"version": 1, "people": {name: {"embeddings": [[...512...], ...],
                                              "sources": ["path", ...]}}}
    """

    def __init__(self, path: Path = GALLERY_PATH):
        self.path = Path(path)
        self.people: dict[str, dict[str, list]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.people = data.get("people", {})

    def save(self) -> None:
        self.path.write_text(json.dumps({"version": 1, "people": self.people}, indent=2))

    def enroll(self, name: str, embedding: np.ndarray, source: str) -> int:
        person = self.people.setdefault(name, {"embeddings": [], "sources": []})
        person["embeddings"].append([float(x) for x in embedding])
        person["sources"].append(source)
        return len(person["embeddings"])

    def remove(self, name: str) -> bool:
        return self.people.pop(name, None) is not None

    def names(self) -> list[str]:
        return sorted(self.people)

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """Return identities ranked by best cosine similarity to `embedding`."""
        results = []
        for name, person in self.people.items():
            sims = [cosine_similarity(embedding, e) for e in person["embeddings"]]
            best = max(sims) if sims else -1.0
            results.append({
                "name": name,
                "similarity": round(best, 4),
                "mean_similarity": round(float(np.mean(sims)), 4) if sims else None,
                "num_samples": len(sims),
            })
        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:top_k]


# --------------------------------------------------------------------------- #
# Drawing / crops (optional output)
# --------------------------------------------------------------------------- #

# Default box/label color (BGR). Keypoints are always drawn in red.
DEFAULT_BOX_COLOR = (0, 200, 0)  # green


def _draw_annotations(img: np.ndarray, records: list[FaceRecord],
                      labels: dict[int, str] | None = None,
                      colors: dict[int, tuple[int, int, int]] | None = None,
                      default_color: tuple[int, int, int] = DEFAULT_BOX_COLOR) -> np.ndarray:
    """Draw bounding boxes, keypoints and labels onto a BGR image (in place).

    `labels` optionally maps a face index to a custom caption (e.g. a matched
    name); otherwise a default "#i gender age" caption is used.
    `colors` optionally maps a face index to a BGR box/label color; faces not in
    the map use `default_color`. Use this to highlight matches (e.g. green) vs.
    non-matches (e.g. red).
    """
    labels = labels or {}
    colors = colors or {}
    for r in records:
        x1, y1, x2, y2 = (int(v) for v in r.bbox)
        color = colors.get(r.index, default_color)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = labels.get(
            r.index, f"#{r.index} {r.gender or '?'} {r.age if r.age is not None else '?'}")
        cv2.putText(img, label, (x1, max(12, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for x, y in r.kps5:
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
    return img


def save_annotated(path: str | Path, records: list[FaceRecord], out_path: Path,
                   labels: dict[int, str] | None = None,
                   colors: dict[int, tuple[int, int, int]] | None = None) -> None:
    """Save a copy of the image with bounding boxes, keypoints and labels drawn."""
    img = _draw_annotations(_imread(path), records, labels, colors)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def save_annotated_array(img: np.ndarray, records: list[FaceRecord], out_path: Path,
                         labels: dict[int, str] | None = None,
                         colors: dict[int, tuple[int, int, int]] | None = None) -> None:
    """Draw annotations on an in-memory BGR array and write it to disk."""
    drawn = _draw_annotations(img.copy(), records, labels, colors)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), drawn)


def annotate_rgb(path: str | Path, records: list[FaceRecord],
                 labels: dict[int, str] | None = None,
                 colors: dict[int, tuple[int, int, int]] | None = None) -> np.ndarray:
    """Return an RGB (H, W, 3) array with annotations drawn — ready for plt.imshow.

    Useful in notebooks: matplotlib expects RGB, OpenCV works in BGR, so this
    draws and converts in one step without touching disk. Pass `colors` to
    highlight specific faces (e.g. matched vs. unmatched).
    """
    return annotate_array_rgb(_imread(path), records, labels, colors)


def annotate_array_rgb(img: np.ndarray, records: list[FaceRecord],
                       labels: dict[int, str] | None = None,
                       colors: dict[int, tuple[int, int, int]] | None = None) -> np.ndarray:
    """Like annotate_rgb but for an in-memory BGR array (e.g. a downloaded image).

    Draws on a copy so the input array is left untouched.
    """
    drawn = _draw_annotations(img.copy(), records, labels, colors)
    return cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# CLI commands
# --------------------------------------------------------------------------- #

def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2))


def cmd_analyze(args) -> int:
    records = analyze_image(args.image)
    if not records:
        _print_json({"image": str(args.image), "faces": []})
        print("No faces detected.", file=sys.stderr)
        return 0
    out = {
        "image": str(args.image),
        "num_faces": len(records),
        "faces": [r.to_dict(include_embedding=args.full,
                            include_dense_landmarks=args.full) for r in records],
    }
    _print_json(out)

    if args.annotate:
        save_annotated(args.image, records, Path(args.annotate))
        print(f"Annotated image saved to {args.annotate}", file=sys.stderr)
    if args.save_embedding:
        np.save(args.save_embedding, np.stack([r.normed_embedding for r in records]))
        print(f"Embeddings saved to {args.save_embedding}", file=sys.stderr)
    return 0


def cmd_verify(args) -> int:
    emb_a = primary_embedding(args.image_a)
    emb_b = primary_embedding(args.image_b)
    sim = cosine_similarity(emb_a, emb_b)
    same = sim >= args.threshold
    _print_json({
        "image_a": str(args.image_a),
        "image_b": str(args.image_b),
        "similarity": round(sim, 4),
        "threshold": args.threshold,
        "same_person": same,
    })
    return 0 if same else 1


def cmd_compare(args) -> int:
    """1:N ad-hoc — compare one probe image against many candidate images/folders."""
    probe_emb = primary_embedding(args.probe)
    candidates = expand_images(args.candidates)

    results = []
    for img in candidates:
        if Path(img) == Path(args.probe):
            continue  # skip comparing the probe against itself
        try:
            records = analyze_image(img)
            if not records:
                raise ValueError(f"No face detected in {img}")
        except (ValueError, FileNotFoundError) as e:
            results.append({"image": str(img), "similarity": None, "error": str(e)})
            continue
        # By default only the most prominent face; --all-faces scans every face
        # so a probe can be found in group photos regardless of face size.
        considered = records if args.all_faces else records[:1]
        scored = [(cosine_similarity(probe_emb, r.normed_embedding), r.index)
                  for r in considered]
        best_sim, best_idx = max(scored, key=lambda t: t[0])
        entry = {
            "image": str(img),
            "similarity": round(best_sim, 4),
            "same_person": best_sim >= args.threshold,
            "num_faces": len(records),
        }
        if args.all_faces:
            entry["matched_face_index"] = best_idx
        results.append(entry)

    # Rank matchable candidates by similarity (errors sink to the bottom).
    results.sort(key=lambda r: (r["similarity"] is not None, r["similarity"] or -1),
                 reverse=True)
    matches = [r for r in results if r.get("same_person")]
    _print_json({
        "probe": str(args.probe),
        "threshold": args.threshold,
        "num_candidates": len(results),
        "num_matches": len(matches),
        "best_match": results[0]["image"] if matches else None,
        "results": results,
    })
    return 0 if matches else 1


def cmd_enroll(args) -> int:
    gallery = Gallery(Path(args.gallery))
    added = 0
    for image in args.images:
        try:
            emb = primary_embedding(image)
        except ValueError as e:
            print(f"Skipping {image}: {e}", file=sys.stderr)
            continue
        count = gallery.enroll(args.name, emb, str(image))
        added += 1
        print(f"Enrolled {image} -> '{args.name}' (now {count} sample(s))", file=sys.stderr)
    gallery.save()
    _print_json({"name": args.name, "added": added,
                 "total_samples": len(gallery.people.get(args.name, {}).get("embeddings", []))})
    return 0 if added else 1


def cmd_search(args) -> int:
    gallery = Gallery(Path(args.gallery))
    if not gallery.people:
        print("Gallery is empty — enroll some faces first.", file=sys.stderr)
        return 1
    records = analyze_image(args.image)
    if not records:
        raise ValueError(f"No face detected in {args.image}")

    # By default identify only the most prominent face; --all-faces identifies
    # every face in the query image (useful for group photos).
    faces = records if args.all_faces else records[:1]
    faces_out = []
    for r in faces:
        res = gallery.search(r.normed_embedding, top_k=args.top)
        best = res[0] if res else None
        match = best if best and best["similarity"] >= args.threshold else None
        faces_out.append({
            "face_index": r.index,
            "bbox": [round(v, 2) for v in r.bbox],
            "match": match["name"] if match else None,
            "results": res,
        })

    any_match = any(f["match"] for f in faces_out)
    if args.all_faces:
        _print_json({
            "image": str(args.image),
            "threshold": args.threshold,
            "num_faces": len(records),
            "faces": faces_out,
        })
    else:
        # Preserve the original flat output shape for the single-face case.
        f0 = faces_out[0]
        _print_json({
            "image": str(args.image),
            "threshold": args.threshold,
            "match": f0["match"],
            "results": f0["results"],
        })
    return 0 if any_match else 1


def cmd_gallery(args) -> int:
    gallery = Gallery(Path(args.gallery))
    if args.remove:
        ok = gallery.remove(args.remove)
        gallery.save()
        print(f"{'Removed' if ok else 'Not found'}: {args.remove}", file=sys.stderr)
        return 0 if ok else 1
    summary = {name: len(p["embeddings"]) for name, p in gallery.people.items()}
    _print_json({"gallery": str(gallery.path), "people": summary,
                 "total": sum(summary.values())})
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="facerec",
        description="Facial recognition toolkit (InsightFace / ArcFace).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("analyze", help="Extract biometric data from an image.")
    pa.add_argument("image")
    pa.add_argument("--full", action="store_true",
                    help="Include full 512-d embedding and dense landmarks in output.")
    pa.add_argument("--annotate", metavar="OUT.jpg",
                    help="Save a copy with boxes/keypoints drawn.")
    pa.add_argument("--save-embedding", metavar="OUT.npy",
                    help="Save normed embeddings as a .npy array.")
    pa.set_defaults(func=cmd_analyze)

    pv = sub.add_parser("verify", help="1:1 — are these two images the same person?")
    pv.add_argument("image_a")
    pv.add_argument("image_b")
    pv.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    pv.set_defaults(func=cmd_verify)

    pc = sub.add_parser("compare",
                        help="1:N ad-hoc — compare one probe against many images/folders.")
    pc.add_argument("probe", help="The query image.")
    pc.add_argument("candidates", nargs="+",
                    help="Candidate images and/or directories to compare against.")
    pc.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    pc.add_argument("--all-faces", action="store_true",
                    help="Match the probe against every face in each candidate "
                         "(not just the largest) — finds the probe in group photos.")
    pc.set_defaults(func=cmd_compare)

    pe = sub.add_parser("enroll", help="Add a named person's image(s) to the gallery.")
    pe.add_argument("name")
    pe.add_argument("images", nargs="+")
    pe.add_argument("--gallery", default=str(GALLERY_PATH))
    pe.set_defaults(func=cmd_enroll)

    ps = sub.add_parser("search", help="1:N — find the closest enrolled person.")
    ps.add_argument("image")
    ps.add_argument("--top", type=int, default=5)
    ps.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    ps.add_argument("--gallery", default=str(GALLERY_PATH))
    ps.add_argument("--all-faces", action="store_true",
                    help="Identify every face in the query image (not just the "
                         "largest) against the gallery.")
    ps.set_defaults(func=cmd_search)

    pg = sub.add_parser("gallery", help="List or modify the gallery.")
    pg.add_argument("--list", action="store_true", help="List enrolled people (default).")
    pg.add_argument("--remove", metavar="NAME", help="Remove a person from the gallery.")
    pg.add_argument("--gallery", default=str(GALLERY_PATH))
    pg.set_defaults(func=cmd_gallery)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
