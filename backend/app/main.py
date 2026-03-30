from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the ML model at startup so the first request is not slow
    from app.models.ai_models import _load_model
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Preloading ML model...")
    _load_model()
    logger.info("ML model ready.")
    yield


app = FastAPI(
    title="gRNA Predictor API",
    description="AI-powered guide RNA designer for CRISPR-Cas9",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # open for local dev; restrict to specific origins in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}
