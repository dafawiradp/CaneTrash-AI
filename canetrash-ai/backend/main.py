"""
main.py
=======
FastAPI application entrypoint for CaneTrash-AI.

Run locally:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.database import dispose_db, init_db
from backend.routers import analytics, system_health, transactions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("canetrash.backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info(
        "Starting %s (env=%s, force_synthetic_ai=%s)",
        settings.app_name,
        settings.app_env,
        settings.force_synthetic_ai,
    )
    await init_db()
    yield
    logger.info("Shutting down %s", settings.app_name)
    await dispose_db()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Autonomous Sugarcane Quality & Trash Inspection System -- "
            "Industrial CV + 3D LiDAR + FastAPI backend + Local SLM agent."
        ),
        version="0.1.0-phase1",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(transactions.router, prefix=settings.api_prefix)
    app.include_router(system_health.router, prefix=settings.api_prefix)
    app.include_router(analytics.router, prefix=settings.api_prefix)

    @app.get("/", tags=["root"])
    async def root() -> dict:
        return {
            "service": settings.app_name,
            "status": "running",
            "docs": "/docs",
            "api_prefix": settings.api_prefix,
        }

    return app


app = create_app()
