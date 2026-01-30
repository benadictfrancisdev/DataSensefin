"""FastAPI application entrypoint.

This module is the production-ready entrypoint for the backend service.
It is designed to run both locally (e.g. with `uvicorn main:app --reload`)
and on platforms like Railway without modification.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server import (
    api_router,
    analysis_router,
    ai_router,
    forecast_router,
    ml_router,
    db,
    shutdown_db_client,
)

# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")  # Local development support; Railway uses env vars

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(title="DataSense Backend", version="1.0.0")


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------

# Base allowed origins for local dev and production frontend
_default_vercel_origin = "https://<vercel-frontend-domain>"

frontend_origins = [
    "http://localhost:5173",
    os.getenv("FRONTEND_ORIGIN", _default_vercel_origin),
]

# Allow overriding completely via CORS_ORIGINS env (comma-separated)
cors_override = os.getenv("CORS_ORIGINS")
if cors_override:
    frontend_origins = [origin.strip() for origin in cors_override.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

app.include_router(api_router)
app.include_router(analysis_router)
app.include_router(ml_router)
app.include_router(ai_router)
app.include_router(forecast_router)


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup() -> None:
    """Perform a lightweight database connectivity check on startup."""
    try:
        # `ping` is cheap and recommended for checking MongoDB connectivity
        await db.command("ping")
        app.state.db_connected = True
        logger.info("Successfully connected to MongoDB.")
    except Exception as exc:  # pragma: no cover - defensive logging
        app.state.db_connected = False
        app.state.db_error = str(exc)
        logger.error("MongoDB connection failed: %s", exc)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Close the shared MongoDB client when the app shuts down."""
    await shutdown_db_client()


# ---------------------------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Basic health-check endpoint.

    Returns overall backend status and database connectivity information.
    """

    db_connected = getattr(app.state, "db_connected", False)
    db_error = getattr(app.state, "db_error", None)

    status = {
        "backend": "ok",
        "database": {
            "connected": db_connected,
            "error": db_error if not db_connected else None,
        },
    }

    return status