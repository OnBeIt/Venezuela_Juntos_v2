"""Audit logging — append a row per noteworthy event, viewable at /admin/logs.

Events are stored in the DB (audit_logs table) so they live on the same Railway
volume as the rest of the data and are accessible from the admin frontend. Each
call also echoes to stdout so events show up in Railway's log viewer too.
"""
import logging

from database import AuditLog

_stdout = logging.getLogger("venezuela_juntos.audit")

# Event-type constants (stable strings used for filtering/grouping in the UI).
USER_REGISTER = "user.register"
USER_LOGIN = "user.login"
USER_LOGIN_FAILED = "user.login_failed"
USER_LOGOUT = "user.logout"
USER_DELETED = "admin.user_deleted"
REPORT_MISSING = "report.missing"
REPORT_FOUND = "report.found"
MATCH_CREATED = "match.created"
QUALITY_FLAGGED = "upload.low_quality"
UPLOAD_REJECTED = "upload.rejected"
ADMIN_DELETE = "admin.delete"


def log_event(session, event_type: str, message: str, *, actor=None, request=None) -> None:
    """Append an audit entry. The caller's session commit persists it.

    `actor` is a User (or None for anonymous/system); `request` supplies the
    client IP when available.
    """
    ip = None
    if request is not None and request.client is not None:
        ip = request.client.host

    session.add(AuditLog(
        event_type=event_type,
        actor_email=(actor.email if actor is not None else None),
        actor_id=(actor.id if actor is not None else None),
        message=message,
        ip=ip,
    ))
    who = actor.email if actor is not None else "anonymous"
    _stdout.info("[%s] %s — %s", event_type, who, message)
