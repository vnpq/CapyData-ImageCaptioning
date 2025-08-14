# app_api.py
import io, time, os
from typing import Optional, List
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from model_adapters import load_captioners

CKPT_DIR = os.getenv("CKPT_DIR", "checkpoints")
MODELS = load_captioners(ckpt_dir=CKPT_DIR)

app = FastAPI(title="VN Sport Captioning API", version="0.1.0")

class CaptionResponse(BaseModel):
    model: str
    captions: List[str]
    latency_ms: float

@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(MODELS.keys())}

@app.get("/models")
def models():
    return {"available": list(MODELS.keys())}

@app.post("/caption", response_model=CaptionResponse)
async def caption(
    file: UploadFile = File(...),
    model: str = Query("vit_t5", description="cnn_lstm | cnn_t5 | vit_t5"),
    num_captions: int = Query(1, ge=1, le=5, description="Số caption cần sinh (tối đa 5)")
):
    if model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'.")
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    t0 = time.time()
    captions = MODELS[model].caption(img, num_captions=num_captions)  # <-- KHÔNG còn max_new_tokens ở API
    return CaptionResponse(model=model, captions=captions, latency_ms=(time.time()-t0)*1000.0)
