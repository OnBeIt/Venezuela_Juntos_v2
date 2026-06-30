"""Authentication: password hashing, cookie sessions, and the admin bootstrap.

Sessions ride on Starlette's SessionMiddleware (a signed cookie), so we only
store the user id in request.session and resolve the User row per request.
Password hashing uses the standard library (PBKDF2-HMAC-SHA256) so there are no
extra native dependencies to build in the container.
"""
import hashlib
import hmac
import os
import secrets

from fastapi import Request

from database import FoundPerson, MissingPerson, User, get_session

_PBKDF2_ROUNDS = 200_000


def hash_password(password: str) -> str:
    """Return a 'salt$hash' string (both hex) for storage."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of a password against a stored 'salt$hash' string."""
    try:
        salt_hex, hash_hex = stored.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ROUNDS)
    return hmac.compare_digest(digest, expected)


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #

def login_user(request: Request, user: User) -> None:
    request.session["user_id"] = user.id


def logout_user(request: Request) -> None:
    request.session.pop("user_id", None)


def current_user(request: Request, session) -> User | None:
    """Resolve the logged-in User from the session, or None."""
    user_id = request.session.get("user_id")
    if user_id is None:
        return None
    return session.get(User, user_id)


# --------------------------------------------------------------------------- #
# Admin bootstrap
# --------------------------------------------------------------------------- #

def ensure_admin() -> None:
    """Create or update the admin account from ADMIN_EMAIL / ADMIN_PASSWORD.

    Also backfills any legacy missing/found rows that predate the auth feature
    (owner_id IS NULL) so they belong to the admin — visible and deletable by
    the admin, invisible to regular users.
    """
    email = os.environ.get("ADMIN_EMAIL", "").strip().lower()
    password = os.environ.get("ADMIN_PASSWORD", "")
    if not email or not password:
        return

    import audit

    with get_session() as session:
        admin = session.query(User).filter(User.email == email).one_or_none()
        created = admin is None
        if created:
            admin = User(email=email, password_hash=hash_password(password), is_admin=True)
            session.add(admin)
        else:
            # Keep the admin account in sync with the configured credentials.
            admin.password_hash = hash_password(password)
            admin.is_admin = True
        session.flush()

        backfilled = 0
        for model in (MissingPerson, FoundPerson):
            backfilled += session.query(model).filter(model.owner_id.is_(None)).update(
                {model.owner_id: admin.id}, synchronize_session=False
            )
        if created:
            audit.log_event(session, audit.USER_REGISTER,
                            f"Admin account created: {email}"
                            + (f" (backfilled {backfilled} legacy report(s))" if backfilled else ""),
                            actor=admin)
        session.commit()
