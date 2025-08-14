import os
from typing import Dict
from PIL import Image
import torch
import torchvision.transforms as T

# đổi tên nếu file là tokenizer_shared.py
from tokenizer import load_shared_tokenizer, token_dicts_from_tokenizer

from my_models.cnn_lstm import build_model_cnn_lstm
from my_models.cnn_t5   import build_model_cnn_t5
from my_models.vit_t5   import build_model_vit_t5

VOCAB_JSON  = os.getenv("VOCAB_JSON", "checkpoints/idx_to_word.json")
T5_PRETRAIN = os.getenv("T5_PRETRAIN", "t5-base")
MAX_LEN     = int(os.getenv("MAX_LEN", "25"))


TOKENIZER = load_shared_tokenizer(VOCAB_JSON, hf_fallback=T5_PRETRAIN, max_len=MAX_LEN)
TOKEN2IDX, IDX2TOKEN = token_dicts_from_tokenizer(TOKENIZER)
VOCAB_SIZE = len(TOKEN2IDX)

def _pp_norm224():
    return T.Compose([
        T.Resize((224,224)), T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

class BaseCaptioner(torch.nn.Module):
    def __init__(self): super().__init__()
    @torch.inference_mode()
    def caption(self, image: Image.Image, max_new_tokens: int = 25) -> str: raise NotImplementedError

class CaptionerCNNLSTM(BaseCaptioner):
    def __init__(self, ckpt_path: str, device: str):
        super().__init__()
        self.model = build_model_cnn_lstm(vocab_size=VOCAB_SIZE).to(device).eval()
        state = torch.load(ckpt_path, map_location="cpu")
        try:
            self.model.load_state_dict(state.get("model", state), strict=False)
        except Exception as e:
            print("[WARN] load_state_dict CNN-LSTM:", e)
        self.pp = _pp_norm224()
        self.device = device
        
    @torch.inference_mode()
    def caption(self, image: Image.Image, num_captions: int = 1) -> list[str]:
        # preprocess
        x = self.pp(image).unsqueeze(0).to(self.device)
        # ưu tiên dùng hàm generate_captions mới (sampling); fallback nếu chưa có
        if hasattr(self.model, "generate_captions"):
            caps = self.model.generate_captions(
                x[0], TOKEN2IDX, IDX2TOKEN,
                max_length=MAX_LEN, num_captions=num_captions,
                top_p=0.9, top_k=50, temperature=0.8
            )
            return caps[:num_captions]
        else:
            # Fallback: gọi nhiều lần hàm cũ (nếu hàm cũ đang greedy thì caption có thể giống nhau)
            outs = []
            for _ in range(num_captions):
                cands = self.model.generate_caption(
                    x[0], TOKEN2IDX, IDX2TOKEN,
                    max_length=MAX_LEN, beam_width=1  # beam=1 để bớt “cứng”
                )
                text = cands[0] if isinstance(cands, list) and cands else str(cands)
                if text and text not in outs:
                    outs.append(text)
            return outs or [""]

class CaptionerCNNT5(BaseCaptioner):
    def __init__(self, ckpt_path: str, device: str):
        super().__init__()
        self.model = build_model_cnn_t5(tokenizer=TOKENIZER, pretrained_model_name=T5_PRETRAIN).to(device).eval()
        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state.get("model", state), strict=False)
        self.device = device
        
    @torch.inference_mode()
    def caption(self, image: Image.Image, num_captions: int = 1):
        # Ưu tiên gọi generate_captions nếu có
        if hasattr(self.model, "generate_captions"):
            return self.model.generate_captions(
                image, num_captions=num_captions, max_length=32,
                temperature=0.8, top_p=0.9, top_k=50
            )
        # Fallback (cũ): chỉ 1 caption
        return [self.model.generate_caption(image, max_length=32)]

class CaptionerViTT5(BaseCaptioner):
    def __init__(self, ckpt_path: str, device: str):
        super().__init__()
        vit = build_model_vit_t5(tokenizer=TOKENIZER, pretrained_model_name=T5_PRETRAIN, d_model=768)
        self.model = vit.to(device).eval()
        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state.get("model", state), strict=False)
        self.device = device
        
    @torch.inference_mode()
    def caption(self, image: Image.Image, num_captions: int = 1) -> list[str]:
        if hasattr(self.model, "generate_captions"):
            return self.model.generate_captions(
                image, num_captions=num_captions, max_length=MAX_LEN,
                temperature=0.8, top_p=0.9, top_k=50
            )

        return [self.model.generate_caption(image, max_length=MAX_LEN)]

def load_captioners(ckpt_dir: str = "checkpoints", device: str = None) -> Dict[str, BaseCaptioner]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "cnn_lstm": CaptionerCNNLSTM(os.path.join(ckpt_dir, "cnn_lstm.pth"), device).to(device),
        "cnn_t5":   CaptionerCNNT5  (os.path.join(ckpt_dir, "cnn_t5.pt"),   device).to(device),
        "vit_t5":   CaptionerViTT5  (os.path.join(ckpt_dir, "vit_t5.pt"),   device).to(device),
    }