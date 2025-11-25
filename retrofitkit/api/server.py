from fastapi import FastAPI  # type: ignore[import]
from fastapi.responses import HTMLResponse, PlainTextResponse  # type: ignore[import]
from fastapi.staticfiles import StaticFiles  # type: ignore[import]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import]
from retrofitkit.core.app import AppContext
from retrofitkit.api.auth import router as auth_router
from retrofitkit.api.routes import router as api_router
from retrofitkit.core.raman_stream import RamanStreamer
from retrofitkit.core.events import EventBus
from retrofitkit.metrics.exporter import Metrics
from retrofitkit.security.headers import SecurityHeadersMiddleware, RateLimitMiddleware
import time
from datetime import datetime, timezone

app = FastAPI(
    title="POLYMORPH-4 Lite",
    version="2.0.0",
    description="AI-Powered Laboratory Automation Platform",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add security middleware (order matters!)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=100)

# CORS configuration (restrictive by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost"],  # Update in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    max_age=3600,
)

ctx = AppContext.load()
bus = EventBus()
raman_streamer = RamanStreamer(ctx, bus)
_start_time = time.time()

from retrofitkit.api.health import router as health_router

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(api_router, prefix="/api", tags=["api"])
app.include_router(health_router, prefix="/api", tags=["health"]) # This mounts /api/health

app.mount(
    "/static",
    StaticFiles(directory="retrofitkit/api/static"),
    name="static",
)

@app.get("/", response_class=HTMLResponse)
async def index():
    with open(
        "retrofitkit/api/static/index.html",
        "r",
        encoding="utf-8",
    ) as f:
        return HTMLResponse(f.read())

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(Metrics.get().render_prom())

@app.on_event("startup")
async def startup_event():
    await raman_streamer.start()

@app.on_event("shutdown")
async def shutdown_event():
    await raman_streamer.stop()

# Legacy root health endpoints redirecting to new API
@app.get("/health")
async def root_health():
    from retrofitkit.api.health import health_check
    return await health_check()

@app.get("/health/ready")
async def root_ready():
    from retrofitkit.api.health import readiness_check
    return await readiness_check()

@app.get("/health/live")
async def root_live():
    from retrofitkit.api.health import liveness_check
    return await liveness_check()
