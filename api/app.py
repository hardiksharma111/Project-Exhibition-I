"""
FastAPI Application Entrypoint
Mounts Static Web Dashboard, REST API routes, and CORS middleware.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from core.config import BASE_DIR
from api.routes import router as api_router

STATIC_DIR = os.path.join(BASE_DIR, "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    print("[Server] Initializing Adaptive Multi-SLM AI Server...")
    os.makedirs(STATIC_DIR, exist_ok=True)
    yield
    # Shutdown actions
    print("[Server] Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Intelligent Multi-Model AI: Adaptive SLM Routing",
        description="Offline multi-SLM routing, dynamic memory orchestration, cross-model context, and MCP server.",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include REST API routes
    app.include_router(api_router, prefix="/api")

    # Mount static files
    if os.path.exists(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def serve_index():
        index_file = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"status": "online", "message": "Static index.html not yet initialized"}

    return app


app = create_app()
