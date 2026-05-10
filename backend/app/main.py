from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router
from app.api.omics_endpoints import router as omics_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import logging
    logger = logging.getLogger(__name__)

    # Preload the XGBoost model
    from app.models.ai_models import _load_model
    logger.info("Preloading ML model...")
    _load_model()
    logger.info("ML model ready.")

    # Preload OmicsCRISPR predictor (lazy — only if data is present)
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # open for local dev; restrict to specific origins in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router,       prefix="/api/v1")
app.include_router(omics_router, prefix="/api/v1")


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}
