import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.core.limiter import limiter
from app.api.endpoints import router
from app.api.omics_endpoints import router as omics_router

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://jana242k4.github.io",
    "https://janas242k4-grna-predictor-api.hf.space"
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import logging
    logger = logging.getLogger(__name__)

    from app.models.ai_models import _load_model
    logger.info("Preloading ML model...")
    _load_model()
    logger.info("ML model ready.")

    try:
        from omics_pipeline.omics_inference import get_predictor
        p = get_predictor()
        p._load()
        logger.info(
            "OmicsCRISPR predictor ready: %d guides, model=%s",
            p.n_guides, p.has_model,
        )
    except Exception as e:
        logger.warning("OmicsCRISPR predictor not loaded: %s", e)

    yield


app = FastAPI(
    title="gRNA Predictor API",
    description="AI-powered guide RNA designer for CRISPR-Cas9",
    version="1.0.0",
    lifespan=lifespan,
)

# Attach rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — wildcard only in local dev, restricted list in production
_is_dev = os.getenv("APP_ENV", "production").lower() in ("development", "dev", "local")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _is_dev else ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"]    = "nosniff"
    response.headers["X-Frame-Options"]           = "DENY"
    response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"]   = "default-src 'self'"
    if "server" in response.headers:
        del response.headers["server"]
    return response


app.include_router(router,       prefix="/api/v1")
app.include_router(omics_router, prefix="/api/v1")


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}
