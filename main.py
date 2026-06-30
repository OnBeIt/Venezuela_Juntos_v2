import asyncio
import os
import secrets
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

# Allow bare imports of sibling modules (matching, database) regardless of
# how uvicorn is invoked (python -m, module path, or direct file).
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, Form, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import or_
from starlette.middleware.sessions import SessionMiddleware

import audit
import auth
import matching
from database import (
    AuditLog, DATA_DIR, FoundPerson, Match, MissingPerson, User, get_session, init_db,
)

_executor = ThreadPoolExecutor(max_workers=2)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Sessions must be signed. In production set SECRET_KEY so cookies survive a
# redeploy; otherwise we generate an ephemeral key (sessions reset on restart).
SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the InsightFace model on startup so the first upload request
    # doesn't time out loading the 280 MB model pack.
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, matching.warmup)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

init_db()
auth.ensure_admin()

UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ctx(request: Request, user: User | None, **extra) -> dict:
    """Common template context including the current user for the nav."""
    base = {
        "request": request,
        "user": user,
        "is_admin": bool(user and user.is_admin),
    }
    base.update(extra)
    return base


def _login_redirect() -> RedirectResponse:
    return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)


def _save_upload(file: UploadFile, prefix: str, data: bytes) -> str:
    ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    filename = f"{prefix}_{uuid.uuid4().hex}{ext}"
    (UPLOADS_DIR / filename).write_bytes(data)
    return filename


def _user_can_view_photo(session, user: User, filename: str) -> bool:
    """A user may view a photo if admin, the owner, or it is matched to them."""
    if user.is_admin:
        return True

    missing = session.query(MissingPerson).filter(MissingPerson.photo_path == filename).one_or_none()
    found = session.query(FoundPerson).filter(FoundPerson.photo_path == filename).one_or_none()
    entry = missing or found
    if entry is None:
        return False
    if entry.owner_id == user.id:
        return True

    # Visible if this entry is matched to something the user owns.
    if missing is not None:
        return (
            session.query(Match)
            .join(FoundPerson, Match.found_id == FoundPerson.id)
            .filter(Match.missing_id == missing.id, FoundPerson.owner_id == user.id)
            .first()
            is not None
        )
    return (
        session.query(Match)
        .join(MissingPerson, Match.missing_id == MissingPerson.id)
        .filter(Match.found_id == found.id, MissingPerson.owner_id == user.id)
        .first()
        is not None
    )


def _delete_photo(filename: str | None) -> None:
    if filename:
        (UPLOADS_DIR / filename).unlink(missing_ok=True)


def _audit(event: str, message: str, *, actor_id: int | None = None, request: Request = None) -> None:
    """Record a single audit event in its own session (for paths without one open)."""
    with get_session() as session:
        actor = session.get(User, actor_id) if actor_id else None
        audit.log_event(session, event, message, actor=actor, request=request)
        session.commit()


# --------------------------------------------------------------------------- #
# Authentication routes
# --------------------------------------------------------------------------- #

@app.get("/register", response_class=HTMLResponse)
async def register_form(request: Request, error: str = ""):
    return templates.TemplateResponse("register.html", _ctx(request, None, error=error))


@app.post("/register")
async def register_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    password2: str = Form(""),
):
    email = email.strip().lower()
    error = ""
    if "@" not in email or "." not in email:
        error = "Please enter a valid email address."
    elif len(password) < 8:
        error = "Password must be at least 8 characters."
    elif password2 and password != password2:
        error = "Passwords do not match."

    if error:
        return templates.TemplateResponse(
            "register.html", _ctx(request, None, error=error),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    with get_session() as session:
        if session.query(User).filter(User.email == email).first() is not None:
            return templates.TemplateResponse(
                "register.html",
                _ctx(request, None, error="An account with that email already exists."),
                status_code=status.HTTP_409_CONFLICT,
            )
        user = User(email=email, password_hash=auth.hash_password(password), is_admin=False)
        session.add(user)
        session.flush()
        audit.log_event(session, audit.USER_REGISTER, f"New account registered: {email}",
                        actor=user, request=request)
        session.commit()
        auth.login_user(request, user)

    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, error: str = ""):
    return templates.TemplateResponse("login.html", _ctx(request, None, error=error))


@app.post("/login")
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    email = email.strip().lower()
    with get_session() as session:
        user = session.query(User).filter(User.email == email).one_or_none()
        if user is None or not auth.verify_password(password, user.password_hash):
            audit.log_event(session, audit.USER_LOGIN_FAILED,
                            f"Failed login attempt for {email}", request=request)
            session.commit()
            return templates.TemplateResponse(
                "login.html",
                _ctx(request, None, error="Incorrect email or password."),
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        audit.log_event(session, audit.USER_LOGIN, f"Logged in: {email}",
                        actor=user, request=request)
        session.commit()
        auth.login_user(request, user)
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/logout")
async def logout(request: Request):
    with get_session() as session:
        user = auth.current_user(request, session)
        if user is not None:
            audit.log_event(session, audit.USER_LOGOUT, f"Logged out: {user.email}",
                            actor=user, request=request)
            session.commit()
    auth.logout_user(request)
    return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)


# --------------------------------------------------------------------------- #
# Authenticated photo serving (replaces the public StaticFiles mount)
# --------------------------------------------------------------------------- #

@app.get("/uploads/{filename}")
async def serve_upload(request: Request, filename: str):
    # Reject path traversal — only a bare filename within UPLOADS_DIR is valid.
    if "/" in filename or "\\" in filename or filename in ("", ".", ".."):
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    with get_session() as session:
        user = auth.current_user(request, session)
        if user is None:
            return _login_redirect()
        if not _user_can_view_photo(session, user, filename):
            return Response(status_code=status.HTTP_404_NOT_FOUND)

    path = (UPLOADS_DIR / filename).resolve()
    if not path.is_file() or path.parent != UPLOADS_DIR.resolve():
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    return FileResponse(path)


# --------------------------------------------------------------------------- #
# Core pages
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    with get_session() as session:
        user = auth.current_user(request, session)
        if user is None:
            return _login_redirect()
        return templates.TemplateResponse("index.html", _ctx(request, user))


@app.get("/report-missing", response_class=HTMLResponse)
async def report_missing_form(request: Request, error: str = ""):
    with get_session() as session:
        user = auth.current_user(request, session)
        if user is None:
            return _login_redirect()
    return templates.TemplateResponse(
        "report_missing.html", _ctx(request, user, error=error, max_mb=MAX_UPLOAD_MB)
    )


@app.post("/report-missing")
async def report_missing_submit(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    photo: UploadFile = File(...),
):
    with get_session() as session:
        user = auth.current_user(request, session)
        if user is None:
            return _login_redirect()
        owner_id = user.id

    data = await photo.read()
    if len(data) > MAX_UPLOAD_BYTES:
        _audit(audit.UPLOAD_REJECTED,
               f"Missing-person upload rejected: too large ({len(data) // (1024 * 1024)} MB > {MAX_UPLOAD_MB} MB)",
               actor_id=owner_id, request=request)
        return templates.TemplateResponse(
            "report_missing.html",
            _ctx(request, None,
                 error=f"Photo is too large (max {MAX_UPLOAD_MB} MB). Please upload a smaller image.",
                 max_mb=MAX_UPLOAD_MB),
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    loop = asyncio.get_event_loop()
    try:
        emb, low_quality, note = await loop.run_in_executor(_executor, matching.analyze, data)
    except ValueError as e:
        _audit(audit.UPLOAD_REJECTED, f"Missing-person upload rejected: {e}",
               actor_id=owner_id, request=request)
        return templates.TemplateResponse(
            "report_missing.html",
            _ctx(request, None, error=str(e), max_mb=MAX_UPLOAD_MB),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    filename = _save_upload(photo, "missing", data)

    with get_session() as session:
        actor = session.get(User, owner_id)
        person = MissingPerson(
            name=name,
            description=description or None,
            photo_path=filename,
            embedding=matching.embedding_to_blob(emb),
            owner_id=owner_id,
            quality_flag=low_quality,
            quality_note=note,
        )
        session.add(person)
        session.flush()

        audit.log_event(session, audit.REPORT_MISSING,
                        f"Reported missing person '{name}' (id={person.id})",
                        actor=actor, request=request)
        if low_quality:
            audit.log_event(session, audit.QUALITY_FLAGGED,
                            f"Low-quality photo on missing report id={person.id} ({note})",
                            actor=actor, request=request)

        pool = session.query(FoundPerson).all()
        pool_pairs = [(fp, fp.embedding) for fp in pool]
        hits = matching.search_pool(emb, pool_pairs)

        for found_person, sim in hits:
            session.add(Match(missing_id=person.id, found_id=found_person.id, similarity=sim))
            audit.log_event(session, audit.MATCH_CREATED,
                            f"Match: missing id={person.id} ↔ found id={found_person.id} "
                            f"({round(sim * 100)}%)",
                            actor=actor, request=request)

        session.commit()

    return RedirectResponse(_matches_redirect(hits, low_quality), status_code=status.HTTP_303_SEE_OTHER)


@app.get("/report-found", response_class=HTMLResponse)
async def report_found_form(request: Request, error: str = ""):
    with get_session() as session:
        user = auth.current_user(request, session)
        if user is None:
            return _login_redirect()
    return templates.TemplateResponse(
        "report_found.html", _ctx(request, user, error=error, max_mb=MAX_UPLOAD_MB)
    )


@app.post("/report-found")
async def report_found_submit(
    request: Request,
    contact_info: str = Form(...),
    notes: str = Form(""),
    photo: UploadFile = File(...),
):
    with get_session() as session:
        user = auth.current_user(request, session)
        if user is None:
            return _login_redirect()
        owner_id = user.id

    data = await photo.read()
    if len(data) > MAX_UPLOAD_BYTES:
        _audit(audit.UPLOAD_REJECTED,
               f"Found-person upload rejected: too large ({len(data) // (1024 * 1024)} MB > {MAX_UPLOAD_MB} MB)",
               actor_id=owner_id, request=request)
        return templates.TemplateResponse(
            "report_found.html",
            _ctx(request, None,
                 error=f"Photo is too large (max {MAX_UPLOAD_MB} MB). Please upload a smaller image.",
                 max_mb=MAX_UPLOAD_MB),
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    loop = asyncio.get_event_loop()
    try:
        emb, low_quality, note = await loop.run_in_executor(_executor, matching.analyze, data)
    except ValueError as e:
        _audit(audit.UPLOAD_REJECTED, f"Found-person upload rejected: {e}",
               actor_id=owner_id, request=request)
        return templates.TemplateResponse(
            "report_found.html",
            _ctx(request, None, error=str(e), max_mb=MAX_UPLOAD_MB),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    filename = _save_upload(photo, "found", data)

    with get_session() as session:
        actor = session.get(User, owner_id)
        person = FoundPerson(
            contact_info=contact_info,
            notes=notes or None,
            photo_path=filename,
            embedding=matching.embedding_to_blob(emb),
            owner_id=owner_id,
            quality_flag=low_quality,
            quality_note=note,
        )
        session.add(person)
        session.flush()

        audit.log_event(session, audit.REPORT_FOUND,
                        f"Reported found person (id={person.id}, contact={contact_info})",
                        actor=actor, request=request)
        if low_quality:
            audit.log_event(session, audit.QUALITY_FLAGGED,
                            f"Low-quality photo on found report id={person.id} ({note})",
                            actor=actor, request=request)

        pool = session.query(MissingPerson).all()
        pool_pairs = [(mp, mp.embedding) for mp in pool]
        hits = matching.search_pool(emb, pool_pairs)

        for missing_person, sim in hits:
            session.add(Match(missing_id=missing_person.id, found_id=person.id, similarity=sim))
            audit.log_event(session, audit.MATCH_CREATED,
                            f"Match: missing id={missing_person.id} ↔ found id={person.id} "
                            f"({round(sim * 100)}%)",
                            actor=actor, request=request)

        session.commit()

    return RedirectResponse(_matches_redirect(hits, low_quality), status_code=status.HTTP_303_SEE_OTHER)


def _matches_redirect(hits, low_quality: bool) -> str:
    params = []
    if not hits:
        params.append("no_match=1")
    if low_quality:
        params.append("quality=low")
    return "/matches" + ("?" + "&".join(params) if params else "")


@app.get("/matches", response_class=HTMLResponse)
async def matches_page(request: Request, no_match: int = 0, quality: str = ""):
    with get_session() as session:
        user = auth.current_user(request, session)
        if user is None:
            return _login_redirect()

        query = (
            session.query(Match)
            .join(MissingPerson, Match.missing_id == MissingPerson.id)
            .join(FoundPerson, Match.found_id == FoundPerson.id)
        )
        if not user.is_admin:
            query = query.filter(
                or_(MissingPerson.owner_id == user.id, FoundPerson.owner_id == user.id)
            )
        matches = query.order_by(Match.matched_at.desc()).all()

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
                "low_quality": bool(m.missing_person.quality_flag or m.found_person.quality_flag),
            }
            for m in matches
        ]

        # The viewer's own uploads that have not matched yet — "match outstanding".
        own_missing = session.query(MissingPerson).filter(MissingPerson.owner_id == user.id).all()
        own_found = session.query(FoundPerson).filter(FoundPerson.owner_id == user.id).all()
        pending = []
        for p in own_missing:
            if not p.matches:
                pending.append({
                    "kind": "missing", "title": p.name, "subtitle": p.description,
                    "photo": p.photo_path, "low_quality": bool(p.quality_flag),
                    "uploaded_at": p.uploaded_at.strftime("%Y-%m-%d %H:%M UTC") if p.uploaded_at else "",
                    "sort": p.uploaded_at,
                })
        for p in own_found:
            if not p.matches:
                pending.append({
                    "kind": "found", "title": p.contact_info, "subtitle": p.notes,
                    "photo": p.photo_path, "low_quality": bool(p.quality_flag),
                    "uploaded_at": p.uploaded_at.strftime("%Y-%m-%d %H:%M UTC") if p.uploaded_at else "",
                    "sort": p.uploaded_at,
                })
        pending.sort(key=lambda r: (r["sort"] is not None, r["sort"]), reverse=True)

        # Personal stats for the viewer (what *they* added + matches involving them).
        my_match_total = (
            session.query(Match)
            .join(MissingPerson, Match.missing_id == MissingPerson.id)
            .join(FoundPerson, Match.found_id == FoundPerson.id)
            .filter(or_(MissingPerson.owner_id == user.id, FoundPerson.owner_id == user.id))
            .count()
        )
        stats = {
            "my_missing": len(own_missing),
            "my_found": len(own_found),
            "my_matches": my_match_total,
            "my_pending": len(pending),
        }

        return templates.TemplateResponse(
            "matches.html",
            _ctx(request, user, matches=rows, pending=pending, stats=stats,
                 no_match=no_match, quality_low=(quality == "low")),
        )


# --------------------------------------------------------------------------- #
# Admin console
# --------------------------------------------------------------------------- #

@app.get("/admin", response_class=HTMLResponse)
async def admin_console(request: Request):
    with get_session() as session:
        user = auth.current_user(request, session)
        if user is None:
            return _login_redirect()
        if not user.is_admin:
            return Response(status_code=status.HTTP_403_FORBIDDEN)

        missing = session.query(MissingPerson).order_by(MissingPerson.uploaded_at.desc()).all()
        found = session.query(FoundPerson).order_by(FoundPerson.uploaded_at.desc()).all()
        matches = session.query(Match).order_by(Match.matched_at.desc()).all()
        users = session.query(User).order_by(User.created_at.asc()).all()

        owners = {u.id: u.email for u in users}

        # Per-user upload counts so the admin sees what a deletion would reassign.
        missing_by_owner, found_by_owner = {}, {}
        for p in missing:
            missing_by_owner[p.owner_id] = missing_by_owner.get(p.owner_id, 0) + 1
        for p in found:
            found_by_owner[p.owner_id] = found_by_owner.get(p.owner_id, 0) + 1

        user_rows = [
            {
                "id": u.id, "email": u.email, "is_admin": bool(u.is_admin),
                "is_self": u.id == user.id,
                "missing_count": missing_by_owner.get(u.id, 0),
                "found_count": found_by_owner.get(u.id, 0),
                "created_at": u.created_at.strftime("%Y-%m-%d %H:%M") if u.created_at else "",
            }
            for u in users
        ]

        missing_rows = [
            {
                "id": p.id, "name": p.name, "photo": p.photo_path,
                "owner": owners.get(p.owner_id, "—"),
                "match_count": len(p.matches),
                "low_quality": bool(p.quality_flag), "quality_note": p.quality_note,
                "uploaded_at": p.uploaded_at.strftime("%Y-%m-%d %H:%M") if p.uploaded_at else "",
            }
            for p in missing
        ]
        found_rows = [
            {
                "id": p.id, "contact_info": p.contact_info, "photo": p.photo_path,
                "owner": owners.get(p.owner_id, "—"),
                "match_count": len(p.matches),
                "low_quality": bool(p.quality_flag), "quality_note": p.quality_note,
                "uploaded_at": p.uploaded_at.strftime("%Y-%m-%d %H:%M") if p.uploaded_at else "",
            }
            for p in found
        ]
        match_rows = [
            {
                "id": m.id,
                "missing_name": m.missing_person.name,
                "missing_photo": m.missing_person.photo_path,
                "found_photo": m.found_person.photo_path,
                "contact_info": m.found_person.contact_info,
                "similarity_pct": round(m.similarity * 100),
                "matched_at": m.matched_at.strftime("%Y-%m-%d %H:%M") if m.matched_at else "",
            }
            for m in matches
        ]

        # Platform-wide stats + what the admin personally added. An entry counts
        # as "matched" if it appears in at least one match.
        matched_missing_ids = {m.missing_id for m in matches}
        matched_found_ids = {m.found_id for m in matches}
        stats = {
            "total_matches": len(matches),
            "total_missing": len(missing),
            "total_found": len(found),
            "total_users": len(users),
            "pending_missing": sum(1 for p in missing if p.id not in matched_missing_ids),
            "pending_found": sum(1 for p in found if p.id not in matched_found_ids),
            "my_missing": missing_by_owner.get(user.id, 0),
            "my_found": found_by_owner.get(user.id, 0),
        }

        return templates.TemplateResponse(
            "admin.html",
            _ctx(request, user, missing=missing_rows, found=found_rows,
                 matches=match_rows, users=user_rows, stats=stats),
        )


# Declared before the generic /admin/delete/{kind}/{id} route so "user" is not
# captured as a {kind}.
@app.post("/admin/delete/user/{user_id}")
async def admin_delete_user(request: Request, user_id: int):
    with get_session() as session:
        user = auth.current_user(request, session)
        if user is None:
            return _login_redirect()
        if not user.is_admin:
            return Response(status_code=status.HTTP_403_FORBIDDEN)

        target = session.get(User, user_id)
        if target is None:
            return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)
        # Guard: an admin cannot delete their own account.
        if target.id == user.id:
            return RedirectResponse("/admin?error=self", status_code=status.HTTP_303_SEE_OTHER)

        # Reassign the user's reports to the admin so no missing-person data is
        # lost; then remove the account.
        reassigned = 0
        for model in (MissingPerson, FoundPerson):
            reassigned += (
                session.query(model)
                .filter(model.owner_id == target.id)
                .update({model.owner_id: user.id}, synchronize_session=False)
            )
        email = target.email
        session.delete(target)
        audit.log_event(session, audit.USER_DELETED,
                        f"Deleted user {email} (reassigned {reassigned} report(s) to {user.email})",
                        actor=user, request=request)
        session.commit()

    return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/admin/delete/{kind}/{entry_id}")
async def admin_delete(request: Request, kind: str, entry_id: int):
    with get_session() as session:
        user = auth.current_user(request, session)
        if user is None:
            return _login_redirect()
        if not user.is_admin:
            return Response(status_code=status.HTTP_403_FORBIDDEN)

        if kind == "missing":
            entry = session.get(MissingPerson, entry_id)
            if entry is not None:
                _delete_photo(entry.photo_path)
                session.delete(entry)  # cascades to its matches
                audit.log_event(session, audit.ADMIN_DELETE,
                                f"Deleted missing person id={entry_id} ('{entry.name}')",
                                actor=user, request=request)
        elif kind == "found":
            entry = session.get(FoundPerson, entry_id)
            if entry is not None:
                _delete_photo(entry.photo_path)
                session.delete(entry)  # cascades to its matches
                audit.log_event(session, audit.ADMIN_DELETE,
                                f"Deleted found person id={entry_id} (contact={entry.contact_info})",
                                actor=user, request=request)
        elif kind == "match":
            entry = session.get(Match, entry_id)
            if entry is not None:
                session.delete(entry)  # removes the pairing only, not the people
                audit.log_event(session, audit.ADMIN_DELETE,
                                f"Deleted match id={entry_id} "
                                f"(missing={entry.missing_id}, found={entry.found_id})",
                                actor=user, request=request)
        else:
            return Response(status_code=status.HTTP_400_BAD_REQUEST)

        session.commit()

    return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/admin/logs", response_class=HTMLResponse)
async def admin_logs(request: Request):
    with get_session() as session:
        user = auth.current_user(request, session)
        if user is None:
            return _login_redirect()
        if not user.is_admin:
            return Response(status_code=status.HTTP_403_FORBIDDEN)

        events = (
            session.query(AuditLog)
            .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
            .limit(500)
            .all()
        )
        rows = [
            {
                "when": e.created_at.strftime("%Y-%m-%d %H:%M:%S") if e.created_at else "",
                "event_type": e.event_type,
                "actor": e.actor_email or "—",
                "message": e.message,
                "ip": e.ip or "—",
            }
            for e in events
        ]
        return templates.TemplateResponse("admin_logs.html", _ctx(request, user, events=rows))
