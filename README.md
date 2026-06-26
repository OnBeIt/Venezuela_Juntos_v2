# Disaster Relief Face Match — CLAUDE.md

## What this project is

A web application for disaster relief scenarios. Families upload photos of missing persons;
rescuers upload photos of people they have found and attach a contact phone number or email.
The app automatically cross-matches faces using ArcFace embeddings and publishes confirmed
matches on a public page so families can reach the rescuer directly.

---

## Repository layout

```
disaster-relief-app/        ← you are here (this is the self-contained app root)
├── CLAUDE.md               ← this file
├── main.py                 ← FastAPI app + all routes
├── database.py             ← SQLAlchemy models + SQLite engine
├── matching.py             ← face embedding + vectorized search wrapper
├── requirements.txt        ← Python dependencies
├── Dockerfile              ← production container (Railway-ready)
├── lib/
│   └── facerec.py          ← ArcFace/InsightFace core (do not modify)
├── templates/
│   ├── base.html           ← Tailwind CDN layout shell + nav
│   ├── index.html          ← home page (two CTA cards)
│   ├── report_missing.html ← family upload form
│   ├── report_found.html   ← rescuer upload form
│   └── matches.html        ← public confirmed-matches grid
└── data/                   ← created at runtime (gitignored)
    ├── db.sqlite            ← SQLite database
    └── uploads/             ← uploaded photos
```

This folder is fully self-contained — you can move it anywhere and it will work
without any other files from the original project.

---

## Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Web framework | FastAPI | server-rendered with Jinja2 |
| Templates | Jinja2 + Tailwind CDN | no build step required |
| Database | SQLite via SQLAlchemy 2.x | single file, persisted on Railway volume |
| Face recognition | InsightFace `buffalo_l` (ArcFace) | ~280 MB model, downloaded once |
| Inference backend | ONNX Runtime | CPU-only (`CPUExecutionProvider`) |
| Image I/O | OpenCV headless | headless build required in containers |

---

## Running locally

```bash
# From inside disaster-relief-app/
pip install -r requirements.txt
uvicorn main:app --reload
# → http://localhost:8000
```

On first startup InsightFace downloads the `buffalo_l` model pack (~280 MB)
to `~/.insightface/`. This only happens once.

Data is stored in `disaster-relief-app/data/` by default. Override with:

```bash
DATA_DIR=/some/other/path uvicorn main:app --reload
```

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `./data` (next to `main.py`) | Root for SQLite DB and uploads |
| `INSIGHTFACE_HOME` | `~/.insightface` | Model cache directory |

In production (Railway) set `DATA_DIR=/data` and mount a persistent volume at `/data`.
Set `INSIGHTFACE_HOME=/data/.insightface` so the model survives container redeploys.

---

## Database schema (SQLite)

Three tables, created automatically by `init_db()` on startup:

**`missing_persons`** — one row per family upload
- `id`, `name` (TEXT), `description` (TEXT nullable), `photo_path` (TEXT),
  `embedding` (BLOB — float32 bytes, 512-d ArcFace vector), `uploaded_at`

**`found_persons`** — one row per rescuer upload
- `id`, `contact_info` (TEXT — phone or email), `notes` (TEXT nullable),
  `photo_path`, `embedding` (BLOB), `uploaded_at`

**`matches`** — join table populated automatically on every upload
- `id`, `missing_id` (FK), `found_id` (FK), `similarity` (FLOAT 0–1), `matched_at`

Photos are stored as files under `{DATA_DIR}/uploads/` with filenames like
`missing_<uuid>.jpg` or `found_<uuid>.jpg`. `photo_path` in the DB is the
filename only (not the full path) and is served at `/uploads/<filename>`.

---

## How matching works

1. A photo is uploaded (family or rescuer).
2. `matching.embed(bytes)` calls `facerec.analyze_bytes()` → returns the 512-d
   L2-normed ArcFace embedding of the largest detected face. Raises `ValueError`
   if no face is found (shown as a form error to the user).
3. All embeddings from the *opposite* pool are loaded from the DB as raw bytes
   and stacked into a NumPy matrix.
4. `matrix @ probe` computes cosine similarity for every stored face in one
   vectorized operation (both sides are already L2-normed → dot product = cosine sim).
5. Any pair with similarity ≥ **0.40** (the ArcFace default threshold in `facerec.py`)
   is inserted into the `matches` table.
6. The `/matches` page shows all confirmed matches sorted by most recent.

To tune the threshold, change `THRESHOLD` in `matching.py` (it reads
`facerec.DEFAULT_MATCH_THRESHOLD` which is `0.40`). Raise it to reduce false
positives; lower it to catch more distant matches.

---

## Routes

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Home page |
| GET | `/report-missing` | Family upload form |
| POST | `/report-missing` | Save photo → embed → search found pool → store → redirect to `/matches` |
| GET | `/report-found` | Rescuer upload form |
| POST | `/report-found` | Save photo → embed → search missing pool → store → redirect to `/matches` |
| GET | `/matches` | Public matches grid |
| GET | `/uploads/<filename>` | Serve uploaded photos (StaticFiles) |

---

## Docker / Railway deployment

Build from inside the `disaster-relief-app/` folder — it is fully self-contained:

```bash
# From inside disaster-relief-app/
docker build -t disaster-relief .
docker run -p 8000:8000 -v $(pwd)/data:/data disaster-relief
```

**Railway setup:**
1. Push the `disaster-relief-app/` folder (or its own repo) to GitHub.
2. Create a new Railway project → Deploy from GitHub repo.
3. Railway detects the Dockerfile automatically.
4. Add a persistent volume mounted at `/data`.
5. Set environment variables: `DATA_DIR=/data`, `INSIGHTFACE_HOME=/data/.insightface`.
6. Railway detects the Dockerfile automatically.

---

## Known limitations / next steps

- **No deduplication:** uploading the same photo twice creates two DB rows and
  may produce a self-match. Add a hash check on upload to prevent this.
- **No pagination:** the `/matches` page loads all matches at once. Add
  limit/offset once the dataset grows.
- **No admin interface:** there is no way to delete records via the UI.
  Access the SQLite file directly, or add a password-protected `/admin` route.
- **Single-face assumption:** only the largest detected face in each photo is
  indexed. Group photos are not supported.
- **No email/SMS notification:** matches are only visible on the public page.
  Integrate SendGrid or Twilio to proactively notify families when a match appears.
- **No HTTPS enforcement:** configure Railway's built-in TLS or add a reverse
  proxy; do not run this over plain HTTP in production.
