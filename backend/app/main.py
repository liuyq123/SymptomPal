import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .routes import ingest, logs, summarize, medications, ambient, checkins, profile, cycle
from .services.safety import EDUCATIONAL_DISCLAIMER
from .services.auth import require_api_key, _authenticate_and_get_bound_user, create_session
from .middleware.rate_limit import RateLimitMiddleware

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and optionally pre-load models on startup."""
    # Initialize database schema once at startup (not per-request)
    from .services.storage import _ensure_db
    _ensure_db()

    if os.environ.get("PRELOAD_HEAR", "").lower() == "true":
        logger.info("Pre-loading HeAR model...")
        try:
            from .services.audio_classifier import get_audio_classifier_client
            client = get_audio_classifier_client()
            client._ensure_initialized()
            logger.info("HeAR model pre-loaded successfully")
        except Exception as e:
            logger.error(f"Failed to pre-load HeAR model: {e}")
    yield

app = FastAPI(
    title="SymptomPal API",
    description="Voice-first symptom tracking with timeline and doctor packet generation",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiting middleware (must be added before CORS so it runs after CORS)
app.add_middleware(RateLimitMiddleware)

# CORS middleware for frontend
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", _default_origins).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (all /api routes require API key)
api_deps = [Depends(require_api_key)]
app.include_router(ingest.router, dependencies=api_deps)
app.include_router(logs.router, dependencies=api_deps)
app.include_router(summarize.router, dependencies=api_deps)
app.include_router(medications.router, dependencies=api_deps)
app.include_router(ambient.router, dependencies=api_deps)
app.include_router(checkins.router, dependencies=api_deps)
app.include_router(profile.router, dependencies=api_deps)
app.include_router(cycle.router, dependencies=api_deps)


@app.get("/")
async def root():
    """Root endpoint with API info and disclaimer."""
    return {
        "name": "SymptomPal API",
        "version": "1.0.0",
        "disclaimer": EDUCATIONAL_DISCLAIMER,
        "endpoints": {
            "ingest_voice": "POST /api/ingest/voice",
            "get_logs": "GET /api/logs?user_id=...",
            "submit_followup": "POST /api/logs/{log_id}/followup",
            "delete_log": "DELETE /api/logs/{log_id}?permanent=false",
            "doctor_packet": "POST /api/summarize/doctor-packet",
            "timeline": "POST /api/summarize/timeline",
            "medications": "GET /api/medications?user_id=...",
            "create_medication": "POST /api/medications",
            "log_medication": "POST /api/medications/log",
            "medication_history": "GET /api/medications/log/history?user_id=...",
            "medication_reminders_pending": "GET /api/medications/reminders/pending?user_id=...",
            "medication_reminder_take": "POST /api/medications/reminders/take",
            "medication_reminder_dismiss": "POST /api/medications/reminders/dismiss",
            "medication_reminder_snooze": "POST /api/medications/reminders/snooze",
            "ambient_start": "POST /api/ambient/sessions/start",
            "ambient_upload": "POST /api/ambient/sessions/upload",
            "ambient_end": "POST /api/ambient/sessions/end",
            "ambient_sessions": "GET /api/ambient/sessions?user_id=...",
            "ambient_active": "GET /api/ambient/sessions/active?user_id=...",
            "ambient_cancel": "POST /api/ambient/sessions/{id}/cancel",
            "checkins_pending": "GET /api/checkins/pending?user_id=...",
            "checkins_respond": "POST /api/checkins/{id}/respond",
            "checkins_dismiss": "POST /api/checkins/{id}/dismiss",
            "profile_get": "GET /api/profile?user_id=...",
            "profile_update": "PATCH /api/profile",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint - verifies database connectivity."""
    try:
        from .services.storage import DB_PATH
        import sqlite3
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("SELECT 1")
        conn.close()
        return {"status": "healthy"}
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "detail": "Database unreachable"},
        )


@app.post("/api/auth/session")
async def create_auth_session(request: Request):
    """
    Create a session cookie bound to a user ID.

    Authentication priority:
    1. X-API-Key header → bind to API key's configured user
    2. X-User-Id header → bind session to requested user (demo/hackathon)

    The session is always bound to a user, preventing impersonation
    via X-User-Id header on subsequent requests.
    """
    api_key = request.headers.get("x-api-key")
    bound_user = None
    if api_key:
        bound_user = _authenticate_and_get_bound_user(api_key)
    if not bound_user:
        bound_user = request.headers.get("x-user-id")
    if not bound_user:
        raise HTTPException(status_code=400, detail="X-User-Id or X-API-Key header required")
    token = create_session(bound_user)
    response = JSONResponse(content={"status": "authenticated"})
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=24 * 3600,
        secure=request.url.scheme == "https",
    )
    return response



# Debug config endpoint removed — exposed model internals without admin-level auth.
