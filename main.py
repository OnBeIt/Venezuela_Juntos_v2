import asyncio
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

# Allow bare imports of sibling modules (matching, database) regardless of
# how uvicorn is invoked (python -m, module path, or direct file).
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, Form, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import matching
from database import DATA_DIR, FoundPerson, Match, MissingPerson, get_session, init_db

_executor = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the InsightFace model on startup so the first upload request
    # doesn't time out loading the 280 MB model pack.
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, matching.warmup)
    yield


app = FastAPI(lifespan=lifespan)
init_db()

UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _save_upload(file: UploadFile, prefix: str) -> tuple[str, bytes]:
    ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    filename = f"{prefix}_{uuid.uuid4().hex}{ext}"
    dest = UPLOADS_DIR / filename
    data = file.file.read()
    dest.write_bytes(data)
    return filename, data


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/report-missing", response_class=HTMLResponse)
async def report_missing_form(request: Request, error: str = ""):
    return templates.TemplateResponse(
        "report_missing.html", {"request": request, "error": error}
    )


@app.post("/report-missing")
async def report_missing_submit(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    photo: UploadFile = File(...),
):
    filename, data = _save_upload(photo, "missing")

    loop = asyncio.get_event_loop()
    try:
        emb = await loop.run_in_executor(_executor, matching.embed, data)
    except ValueError as e:
        (UPLOADS_DIR / filename).unlink(missing_ok=True)
        return templates.TemplateResponse(
            "report_missing.html",
            {"request": request, "error": str(e)},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    with get_session() as session:
        person = MissingPerson(
            name=name,
            description=description or None,
            photo_path=filename,
            embedding=matching.embedding_to_blob(emb),
        )
        session.add(person)
        session.flush()

        pool = session.query(FoundPerson).all()
        pool_pairs = [(fp, fp.embedding) for fp in pool]
        hits = matching.search_pool(emb, pool_pairs)

        for found_person, sim in hits:
            session.add(Match(
                missing_id=person.id,
                found_id=found_person.id,
                similarity=sim,
            ))

        session.commit()

    if not hits:
        return RedirectResponse("/matches?no_match=1", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse("/matches", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/report-found", response_class=HTMLResponse)
async def report_found_form(request: Request, error: str = ""):
    return templates.TemplateResponse(
        "report_found.html", {"request": request, "error": error}
    )


@app.post("/report-found")
async def report_found_submit(
    request: Request,
    contact_info: str = Form(...),
    notes: str = Form(""),
    photo: UploadFile = File(...),
):
    filename, data = _save_upload(photo, "found")

    loop = asyncio.get_event_loop()
    try:
        emb = await loop.run_in_executor(_executor, matching.embed, data)
    except ValueError as e:
        (UPLOADS_DIR / filename).unlink(missing_ok=True)
        return templates.TemplateResponse(
            "report_found.html",
            {"request": request, "error": str(e)},
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    with get_session() as session:
        person = FoundPerson(
            contact_info=contact_info,
            notes=notes or None,
            photo_path=filename,
            embedding=matching.embedding_to_blob(emb),
        )
        session.add(person)
        session.flush()

        pool = session.query(MissingPerson).all()
        pool_pairs = [(mp, mp.embedding) for mp in pool]
        hits = matching.search_pool(emb, pool_pairs)

        for missing_person, sim in hits:
            session.add(Match(
                missing_id=missing_person.id,
                found_id=person.id,
                similarity=sim,
            ))

        session.commit()

    if not hits:
        return RedirectResponse("/matches?no_match=1", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse("/matches", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/matches", response_class=HTMLResponse)
async def matches_page(request: Request, no_match: int = 0):
    with get_session() as session:
        matches = (
            session.query(Match)
            .join(MissingPerson, Match.missing_id == MissingPerson.id)
            .join(FoundPerson, Match.found_id == FoundPerson.id)
            .order_by(Match.matched_at.desc())
            .all()
        )
        rows = [
            {
                "missing_name": m.missing_person.name,
                "missing_photo": m.missing_person.photo_path,
                "missing_description": m.missing_person.description,
                "found_photo": m.found_person.photo_path,
                "found_notes": m.found_person.notes,
                "contact_info": m.found_person.contact_info,
                "similarity_pct": round(m.similarity * 100),
                "matched_at": m.matched_at.strftime("%Y-%m-%d %H:%M UTC"),
            }
            for m in matches
        ]

    return templates.TemplateResponse("matches.html", {"request": request, "matches": rows, "no_match": no_match})
