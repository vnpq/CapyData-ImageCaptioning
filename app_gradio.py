import io
import os
import json
import requests
import gradio as gr
from PIL import Image

API_URL = os.getenv("CAPTION_API_URL", "http://127.0.0.1:8000")

def fetch_models():
    try:
        r = requests.get(f"{API_URL}/models", timeout=10)
        r.raise_for_status()
        data = r.json()
        models = data.get("available", [])
        # đảm bảo có ít nhất 1 lựa chọn
        return models if models else ["vit_t5"]
    except Exception as e:
        print(f"[WARN] Cannot fetch models from API: {e}")
        return ["vit_t5"]

# lấy danh sách model lúc khởi động
MODEL_CHOICES = fetch_models()
DEFAULT_MODEL = "vit_t5" if "vit_t5" in MODEL_CHOICES else MODEL_CHOICES[0]

def refresh_models():
    """Handler cho nút Refresh models: lấy lại danh sách model từ API và cập nhật Radio choices."""
    models = fetch_models()
    # Trả về (choices, value) để gradio cập nhật component Radio động
    default = "vit_t5" if "vit_t5" in models else (models[0] if models else None)
    return gr.update(choices=models, value=default), f"Loaded {len(models)} models from API."

def infer(img: Image.Image, model_name: str, num_captions: int):
    if img is None:
        return "Vui lòng tải ảnh."
    try:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        files = {"file": ("image.png", buf, "image/png")}
        params = {
            "model": model_name,
            "num_captions": int(num_captions)
        }
        r = requests.post(f"{API_URL}/caption", files=files, params=params, timeout=90)
        r.raise_for_status()
        payload = r.json()  
        captions = payload.get("captions", [])
        latency = payload.get("latency_ms", None)

        if not captions:
            return "API trả về rỗng (không có captions)."

        caption_lines = "\n".join(f"{i+1}. {c}" for i, c in enumerate(captions))
        if latency is not None:
            return f"{caption_lines}\n\n⏱ Latency: {latency:.1f} ms"
        return caption_lines

    except requests.HTTPError as he:
        try:
            detail = r.json().get("detail")
        except Exception:
            detail = None
        msg = f"Lỗi HTTP từ API: {he}"
        if detail:
            msg += f"\nChi tiết: {detail}"
        return msg
    except Exception as e:
        return f"Lỗi kết nối hoặc xử lý: {e}"

with gr.Blocks(title="Sport Image Captioning (VN) via API") as demo:
    gr.Markdown(
    f"""
    # Sport Image Captioning (VN)
    Ứng dụng gọi **API** tại `{API_URL}` để sinh caption thay vì chạy model nội bộ.

    - Endpoint dùng: `POST /caption`
    - Tham số: `model`, `num_captions`  
    - Upload ảnh: multipart/form-data (trường `file`)
        """
    )
    with gr.Row():
        img_in = gr.Image(type="pil", label="Upload ảnh")
        with gr.Column():
            model_radio = gr.Radio(
                choices=MODEL_CHOICES,
                value=DEFAULT_MODEL,
                label="Model (từ API /models)"
            )
            refresh_btn = gr.Button("Refresh models", variant="secondary")
            refresh_status = gr.Markdown("", elem_id="refresh-status")

            num_captions = gr.Slider(1, 5, value=3, step=1, label="Số caption (tối đa 5)")
            run_btn = gr.Button("Generate captions")

    out_box = gr.Textbox(label="Captions (vi)", lines=10)

    run_btn.click(
        fn=infer,
        inputs=[img_in, model_radio, num_captions],
        outputs=out_box
    )

    refresh_btn.click(
        fn=refresh_models,
        inputs=[],
        outputs=[model_radio, refresh_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)