import io
import time
import logging
from typing import List

import uvicorn
import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- Import từ vit_t5.py ----
from model.vit_t5 import load_model_from_checkpoint

# Config 
CKPT_PATH  = "checkpoints/vit_t5.pt"
VOCAB_JSON = "data/capydata_ic/vocab/idx_to_word.json"
D_MODEL    = 768
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_MB     = 20
DEBUG      = False

# Logging 
logger = logging.getLogger("api")
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)

# Load model 
MODEL = None
MODEL_NAME = "vit_t5"
try:
    MODEL, _ = load_model_from_checkpoint(
        ckpt_path=CKPT_PATH,
        vocab_json_path=VOCAB_JSON,
        d_model=D_MODEL,
        device=DEVICE,
    )
    MODEL = MODEL.to(DEVICE).eval()
    logger.info("Loaded model from %s on %s", CKPT_PATH, DEVICE)
except Exception:
    logger.exception("Failed to load model from checkpoint")
    MODEL = None

# FastAPI 
app = FastAPI(title="VN Sport Captioning API", version="1.1.0")

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
        "caption": "/caption",
        "device": DEVICE,
        "model_name": MODEL_NAME if MODEL else None,
        "ckpt_path": CKPT_PATH,
        "vocab_json": VOCAB_JSON,
        "ready": MODEL is not None,
    }

@app.get("/health")
def health():
    return {
        "status": "healthy" if MODEL is not None else "degraded",
        "model_name": MODEL_NAME if MODEL else None,
        "device": DEVICE,
    }

@app.post("/caption", response_model=CaptionResponse)
async def caption(
    file: UploadFile = File(..., description="image/*"),
    num_captions: int = Query(1, ge=1, le=5, description="Số caption cần sinh (1–5)")
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh (JPEG/PNG)")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File rỗng")
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File quá lớn (>{MAX_MB}MB)")

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Dữ liệu ảnh không hợp lệ")

    num_beams = max(4, num_captions)
    t0 = time.time()
    try:
        captions = MODEL.generate_captions(
            image_input=img,
            num_captions=num_captions,
            max_length=25,
            min_length=15,
            repetition_penalty=1.3,
            length_penalty=1.0,
            use_beam_search=True,
            num_beams=num_beams,
        )
    except Exception:
        logger.exception("Caption generation failed")
        raise HTTPException(status_code=500, detail="Caption generation failed")

    if not isinstance(captions, list):
        captions = [str(captions)]
    captions = [str(c).strip() for c in captions if str(c).strip()]
    if not captions:
        captions = ["Không thể tạo mô tả cho ảnh này"]

    latency = round((time.time() - t0) * 1000.0, 2)  # tính latency ms
    return CaptionResponse(model=MODEL_NAME, captions=captions, latency_ms=latency)

# Entry 
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=DEBUG)