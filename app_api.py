import io
import os
import time
import logging
from typing import List

import uvicorn

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_adapters import load_captioners


CKPT_DIR = os.getenv("CKPT_DIR", "checkpoints")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

try:
    MODELS = load_captioners(ckpt_dir=CKPT_DIR)
    logger.info("Loaded models: %s", list(MODELS.keys()))
except Exception:
    logger.exception("Failed to load models")
    MODELS = {}

# FastAPI app setup
app = FastAPI(title="VN Sport Captioning API", version="1.0.0")

# CORS middleware to allow frontend apps to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema for caption response
class CaptionResponse(BaseModel):
    model: str
    captions: List[str]
    latency_ms: float


# Endpoints
@app.get("/")
def root():
    return {
        "message": "VN Sport Image Captioning API",
        "health": "/health",
        "models": "/models",
        "caption": "/caption",
        "loaded_models": list(MODELS.keys()),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if MODELS else "degraded",
        "loaded_models": list(MODELS.keys()),
    }


@app.get("/models")
def models():
    return {"available": list(MODELS.keys())}


@app.post("/caption", response_model=CaptionResponse)
async def caption(
    file: UploadFile = File(..., description="image/*"),
    model: str = Query("vit_t5", description="cnn_lstm | cnn_t5 | vit_t5"),
    num_captions: int = Query(1, ge=1, le=5, description="Số caption cần sinh (1–5)"),
):
    # Validate model
    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded")
    if model not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(MODELS.keys())}",
        )

    # Validate file content-type & size
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG)")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_MB}MB)")

    # Decode image
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Inference
    t0 = time.time()
    try:
        caps = MODELS[model].caption(img, num_captions=num_captions)
    except Exception:
        logger.exception("Caption generation failed")
        raise HTTPException(status_code=500, detail="Caption generation failed")

    # Normalize output
    if not isinstance(caps, list):
        caps = [str(caps)]
    captions = [str(c).strip() for c in caps if str(c).strip()]
    if not captions:
        captions = ["Không thể tạo mô tả cho ảnh này"]

    latency = round((time.time() - t0) * 1000.0, 2)
    return CaptionResponse(model=model, captions=captions, latency_ms=latency)


if __name__ == "__main__":
    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=True)