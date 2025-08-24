# -*- coding: utf-8 -*-
import io
import time
import logging
from typing import List, Optional

import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

#  Import từ vit_t5.py (đã có trong repo) 
from model.vit_t5 import load_checkpoint, generate_n_captions   

#  Cấu hình cơ bản 
CKPT_PATH: str  = "checkpoints/vit_t5.pt"
VOCAB_JSON: str = "dataset/capydata_ic/vocab/idx_to_word.json"
DEVICE: str     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_UPLOAD_MB   = 20
MODEL_NAME      = "vit_t5"

#  Logging 
logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

#  Tải model khi khởi động 
MODEL = None
TOKENIZER = None
try:
    MODEL, TOKENIZER = load_checkpoint(
        ckpt_path=CKPT_PATH,
        vocab_path=VOCAB_JSON,
        d_model=768,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
        device=DEVICE,
    )
    MODEL.to(DEVICE).eval()
    logger.info("Loaded checkpoint: %s (device=%s)", CKPT_PATH, DEVICE)
except Exception as e:
    logger.exception("Failed to load checkpoint: %s", e)
    MODEL, TOKENIZER = None, None

#  FastAPI app 
app = FastAPI(title="VN Image Captioning API", version="2.0.0")

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

class HealthResponse(BaseModel):
    status: str
    name: Optional[str]
    device: str
    ready: bool

@app.get("/", response_model=HealthResponse)
def root():
    return HealthResponse(
        status="healthy" if MODEL is not None else "degraded",
        name=MODEL_NAME if MODEL else None,  
        device=DEVICE,
        ready=MODEL is not None,
    )
    
@app.get("/health", response_model=HealthResponse)
def health():
    return root()

@app.post("/caption", response_model=CaptionResponse)
async def caption(
    file: UploadFile = File(..., description="image/*"),
    n: int = Query(5, ge=1, le=10, description="Số caption cần sinh"),
    max_length: int = Query(32, ge=8, le=128),
    num_beams: int = Query(8, ge=1, le=32, description="Beam width (beam search)"),
    num_beam_groups: Optional[int] = Query(4, ge=1, le=32, description="Số nhóm beam để đa dạng"),
    diversity_penalty: float = Query(0.7, ge=0.0, le=10.0),
    repetition_penalty: float = Query(1.1, ge=0.5, le=3.0),
    dedup: bool = Query(True, description="Loại bỏ caption trùng lặp"),
):
    """
    Sinh N captions cho một ảnh bằng beam search đa dạng.
    Lưu ý: Nếu num_beams < n thì tự động nâng lên = n để tránh lỗi.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng")

    # Đọc file ảnh
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File rỗng")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File quá lớn (>{MAX_UPLOAD_MB}MB)")

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Dữ liệu ảnh không hợp lệ")

    # Đảm bảo num_beams >= n
    if num_beams < n:
        num_beams = n

    # Gọi hàm sinh caption
    t0 = time.time()
    try:
        captions = generate_n_captions(
            model=MODEL,
            image_input=img,   
            n=n,
            max_length=max_length,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            repetition_penalty=repetition_penalty,
            dedup=dedup,
        )
    except Exception as e:
        logger.exception("Caption generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {str(e)}")

    latency = round((time.time() - t0) * 1000.0, 2)
    return CaptionResponse(
        model=MODEL_NAME,
        captions=captions,
        latency_ms=latency,
    )