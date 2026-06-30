# Venezuela Juntos

## What this project is

A web application for disaster relief scenarios. Families upload photos of missing persons;
rescuers upload photos of people they have found and attach a contact phone number or email.
The app automatically cross-matches faces using ArcFace embeddings.

The site is **gated by authentication** (self-signup with email + password). Each account only
sees matches that involve its *own* uploads — a missing person it reported, or a found person it
reported. An **admin** account (configured via env vars) sees everything and can delete any
entry, matched or unmatched.

---

## Repository layout

```
disaster-relief-app/        ← you are here (this is the self-contained app root)
├── README.md               ← this file
├── main.py                 ← FastAPI app + all routes
├── auth.py                 ← password hashing, cookie sessions, admin bootstrap
├── audit.py                ← activity-log helper (writes audit_logs + stdout)
├── database.py             ← SQLAlchemy models + SQLite engine
├── matching.py             ← face embedding + quality check + vectorized search
├── requirements.txt        ← Python dependencies
├── Dockerfile              ← production container (Railway-ready)
├── lib/
│   └── facerec.py          ← ArcFace/InsightFace core (do not modify)
├── templates/
│   ├── base.html           ← Tailwind CDN layout shell + auth-aware nav
│   ├── index.html          ← home page (two CTA cards)
│   ├── login.html          ← login form
│   ├── register.html       ← self-signup form
│   ├── report_missing.html ← family upload form
│   ├── report_found.html   ← rescuer upload form
│   ├── matches.html        ← owner-scoped confirmed-matches grid
│   ├── admin.html          ← admin console (entries, matches, users + delete)
│   └── admin_logs.html     ← admin activity-log viewer
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
| `SECRET_KEY` | random per boot | Signs session cookies. **Set this in production** so logins survive redeploys. |
| `ADMIN_EMAIL` | — | Email of the admin account, created/updated on startup. |
| `ADMIN_PASSWORD` | — | Password for the admin account. Set together with `ADMIN_EMAIL`. |
| `MAX_UPLOAD_MB` | `10` | Maximum accepted photo size in megabytes. |

In production (Railway) set `DATA_DIR=/data` and mount a persistent volume at `/data`.
Set `INSIGHTFACE_HOME=/data/.insightface` so the model survives container redeploys.
Set a fixed `SECRET_KEY` and the `ADMIN_EMAIL` / `ADMIN_PASSWORD` pair so you have an admin login.

### Authentication & access model

- The whole site requires login; anonymous visitors are redirected to `/login`.
- Anyone can self-register at `/register` (email + password, min 8 chars). There is **no email
  verification** — no mail server is configured.
- A regular user only sees matches where one side is *their own* upload. Photos are served through
  an authenticated `/uploads/<filename>` route that enforces the same visibility rule, so files
  cannot be fetched by guessing filenames.
- On startup the admin account is created/updated from `ADMIN_EMAIL` / `ADMIN_PASSWORD`, and any
  pre-existing (ownerless) entries are reassigned to it. The admin sees everything at `/admin` and
  can delete any missing/found entry (cascading its matches and removing the photo file) or unlink
  an individual match.
- The admin can also **manage users** from `/admin`: see every account (with its report counts)
  and delete one. Deleting a user **reassigns their reports to the admin** (no data is lost) and
  then removes the account. An admin cannot delete their own account.

### Activity log

Noteworthy events are recorded to the `audit_logs` table and viewable by the admin at
`/admin/logs` (newest first, last 500, with a client-side filter). Logged events include account
registration, login (success **and** failure), logout, missing/found reports, each match created,
low-quality-photo flags, rejected uploads (too large / no face), and every admin deletion (entry,
match, or user). Each entry stores the timestamp, event type, actor email, a human-readable
message, and the client IP. Events are also echoed to stdout so they appear in Railway's log
viewer.

---

## Database schema (SQLite)

Five tables, created automatically by `init_db()` on startup. `init_db()` also runs an
idempotent, guarded migration that `ALTER TABLE ... ADD COLUMN`s the auth/quality columns onto an
existing database (so the live Railway volume upgrades in place — `create_all` alone does not add
columns to pre-existing tables).

**`users`** — one row per account
- `id`, `email` (TEXT unique), `password_hash` (TEXT — `salt$hash`, PBKDF2-HMAC-SHA256),
  `is_admin` (BOOL), `created_at`

**`missing_persons`** — one row per family upload
- `id`, `name` (TEXT), `description` (TEXT nullable), `photo_path` (TEXT),
  `embedding` (BLOB — float32 bytes, 512-d ArcFace vector), `owner_id` (FK → users),
  `quality_flag` (BOOL), `quality_note` (TEXT nullable), `uploaded_at`

**`found_persons`** — one row per rescuer upload
- `id`, `contact_info` (TEXT — phone or email), `notes` (TEXT nullable),
  `photo_path`, `embedding` (BLOB), `owner_id` (FK → users),
  `quality_flag` (BOOL), `quality_note` (TEXT nullable), `uploaded_at`

**`matches`** — join table populated automatically on every upload
- `id`, `missing_id` (FK), `found_id` (FK), `similarity` (FLOAT 0–1), `matched_at`

**`audit_logs`** — append-only activity log shown at `/admin/logs`
- `id`, `created_at`, `event_type` (TEXT), `actor_email` (TEXT nullable), `actor_id` (INT nullable),
  `message` (TEXT), `ip` (TEXT nullable)

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

### Photo quality flag

On upload, `matching.analyze()` reads the quality estimates `facerec` already produces for the
largest face (`blur_var`, `face_px`, `brightness`) and flags the photo as low quality if it is
blurry, the face is too small, or it is too dark / overexposed. Low-quality uploads are **still
accepted and matched** (warn-but-allow); the flag is stored on the entry, a warning banner is
shown after upload, and a "low photo quality" badge appears on the match card. Tune the cutoffs
via the `MIN_BLUR_VAR` / `MIN_FACE_PX` / `MIN_BRIGHTNESS` / `MAX_BRIGHTNESS` constants in
`matching.py`.

---

## Routes

All routes except `/login` and `/register` require a logged-in session.

| Method | Path | What it does |
|---|---|---|
| GET | `/register` | Self-signup form |
| POST | `/register` | Create account (email + password) → log in → `/` |
| GET | `/login` | Login form |
| POST | `/login` | Verify credentials → set session → `/` |
| POST | `/logout` | Clear session → `/login` |
| GET | `/` | Home page |
| GET | `/report-missing` | Family upload form |
| POST | `/report-missing` | Size-check → embed + quality → search found pool → store (owned) → `/matches` |
| GET | `/report-found` | Rescuer upload form |
| POST | `/report-found` | Size-check → embed + quality → search missing pool → store (owned) → `/matches` |
| GET | `/matches` | Matches grid, scoped to the current user (admin sees all) |
| GET | `/uploads/<filename>` | Serve a photo **only** if the user is admin, owns it, or is matched to it |
| GET | `/admin` | Admin console: all entries, matches, and users (admin only) |
| POST | `/admin/delete/{kind}/{id}` | Delete a `missing` / `found` / `match` entry (admin only) |
| POST | `/admin/delete/user/{id}` | Delete a user, reassigning their reports to admin (admin only) |
| GET | `/admin/logs` | Activity log viewer (admin only) |

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

- **No email verification:** self-signup accepts any email without confirming it (no mail server
  is configured). Add SMTP + a verification step before relying on email as identity.
- **No deduplication:** uploading the same photo twice creates two DB rows and
  may produce a self-match. Add a hash check on upload to prevent this.
- **No pagination:** the `/matches` and `/admin` pages load all rows at once. Add
  limit/offset once the dataset grows.
- **Single-face assumption:** only the largest detected face in each photo is
  indexed. Group photos are not supported.
- **No email/SMS notification:** matches are only visible on the (now per-user) matches page.
  Integrate SendGrid or Twilio to proactively notify families when a match appears.
- **No HTTPS enforcement:** configure Railway's built-in TLS or add a reverse
  proxy; do not run this over plain HTTP in production (session cookies depend on it).
