import gradio as gr
from PIL import Image
from model_adapters import load_captioners
import os

CKPT_DIR = os.getenv("CKPT_DIR", "checkpoints")
MODELS = load_captioners(ckpt_dir=CKPT_DIR)

def infer(img: Image.Image, model_name: str, num_captions: int):
    caps = MODELS[model_name].caption(img, num_captions=int(num_captions))  # <-- không còn max_new_tokens
    # Hiển thị gọn theo dạng danh sách
    return "\n".join(f"{i+1}. {c}" for i, c in enumerate(caps))

demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(type="pil", label="Upload ảnh"),
        gr.Radio(choices=list(MODELS.keys()), value="vit_t5", label="Model"),
        gr.Slider(1, 5, value=3, step=1, label="Số caption (tối đa 5)")
    ],
    outputs=gr.Textbox(label="Captions (vi)", lines=8),
    title="Sport Image Captioning (VN)",
    description="Sinh nhiều caption cho 1 ảnh (cnn_lstm, cnn_t5, vit_t5)"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
