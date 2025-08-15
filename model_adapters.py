import os
from typing import Dict, List

import torch
import torchvision.transforms as T
from PIL import Image

from tokenizer import load_shared_tokenizer, token_dicts_from_tokenizer

from my_models import (
    build_model_cnn_lstm,
    build_model_cnn_t5,
    build_model_vit_t5,
)

# Config (env override được)
VOCAB_JSON  = os.getenv("VOCAB_JSON", "data/capydata_ic/vocab/idx_to_word.json")
T5_PRETRAIN = os.getenv("T5_PRETRAIN", "t5-base")
MAX_LEN     = int(os.getenv("MAX_LEN", "25"))

# Load tokenizer dùng chung
TOKENIZER = load_shared_tokenizer(VOCAB_JSON, hf_fallback=T5_PRETRAIN, max_len=MAX_LEN)
TOKEN2IDX, IDX2TOKEN = token_dicts_from_tokenizer(TOKENIZER)
VOCAB_SIZE = len(TOKEN2IDX)


def _pp_norm224():
    """Chuẩn hoá ImageNet cho tensor 224x224 (dùng cho CNN-LSTM)."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class BaseCaptioner(torch.nn.Module):
    """Interface chung để .caption(image, num_captions)"""
    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def caption(self, image: Image.Image, num_captions: int = 1) -> List[str]:
        raise NotImplementedError

    def to(self, device):
        super().to(device)
        self.device = device
        return self


class CaptionerCNNLSTM(BaseCaptioner):
    def __init__(self, ckpt_path: str, device: str):
        super().__init__()
        self.device = device
        self.model = build_model_cnn_lstm(vocab_size=VOCAB_SIZE)
        # nạp checkpoint (nếu có)
        if os.path.exists(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    self.model.load_state_dict(state["model"], strict=False)
                else:
                    self.model.load_state_dict(state, strict=False)
                print(f"[INFO] Loaded CNN-LSTM checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"[WARN] Load CNN-LSTM checkpoint failed: {e}")
        else:
            print(f"[WARN] CNN-LSTM checkpoint not found: {ckpt_path}")

        self.model = self.model.to(device).eval()
        self.pp = _pp_norm224()

    @torch.inference_mode()
    def caption(self, image: Image.Image, num_captions: int = 1) -> List[str]:
        x = self.pp(image).unsqueeze(0).to(self.device)  # (1,3,224,224)

        # prefer generate_captions
        if hasattr(self.model, "generate_captions"):
            try:
                caps = self.model.generate_captions(
                    x, TOKEN2IDX, IDX2TOKEN,
                    max_length=MAX_LEN,
                    num_captions=num_captions,
                    top_p=0.9, top_k=50, temperature=0.8,
                )
                return caps[:num_captions] if caps else [""]
            except Exception as e:
                print(f"[WARN] generate_captions failed (cnn_lstm): {e}")

        # fallback single caption
        if hasattr(self.model, "generate_caption"):
            try:
                cap = self.model.generate_caption(
                    x, TOKEN2IDX, IDX2TOKEN, max_length=MAX_LEN
                )
                if isinstance(cap, list):
                    cap = cap[0] if cap else ""
                return [cap] * num_captions if cap else [""]
            except Exception as e:
                print(f"[WARN] generate_caption failed (cnn_lstm): {e}")

        return [""] * num_captions


class CaptionerCNNT5(BaseCaptioner):
    def __init__(self, ckpt_path: str, device: str):
        super().__init__()
        self.device = device
        self.model = build_model_cnn_t5(
            tokenizer=TOKENIZER,
            pretrained_model_name=T5_PRETRAIN,
        )
        # nạp checkpoint (nếu có)
        if os.path.exists(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    self.model.load_state_dict(state["model"], strict=False)
                else:
                    self.model.load_state_dict(state, strict=False)
                print(f"[INFO] Loaded CNN-T5 checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"[WARN] Load CNN-T5 checkpoint failed: {e}")
        else:
            print(f"[WARN] CNN-T5 checkpoint not found: {ckpt_path}")

        self.model = self.model.to(device).eval()

    @torch.inference_mode()
    def caption(self, image: Image.Image, num_captions: int = 1) -> List[str]:
        try:
            if hasattr(self.model, "generate_captions"):
                return self.model.generate_captions(
                    image,
                    num_captions=num_captions,
                    max_length=MAX_LEN,
                    repetition_penalty=1.2,
                    num_beams=3,
                )
            elif hasattr(self.model, "generate_caption"):
                cap = self.model.generate_caption(image, max_length=MAX_LEN)
                return [cap] * num_captions if cap else [""]
        except Exception as e:
            print(f"[ERROR] CNN-T5 caption failed: {e}")
        return [""] * num_captions


class CaptionerViTT5(BaseCaptioner):
    def __init__(self, ckpt_path: str, device: str):
        super().__init__()
        self.device = device
        self.model = build_model_vit_t5(
            tokenizer=TOKENIZER,
            pretrained_model_name=T5_PRETRAIN,
            d_model=768,
        )
        # nạp checkpoint (nếu có)
        if os.path.exists(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    self.model.load_state_dict(state["model"], strict=False)
                else:
                    self.model.load_state_dict(state, strict=False)
                print(f"[INFO] Loaded ViT-T5 checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"[WARN] Load ViT-T5 checkpoint failed: {e}")
        else:
            print(f"[WARN] ViT-T5 checkpoint not found: {ckpt_path}")

        self.model = self.model.to(device).eval()

    @torch.inference_mode()
    def caption(self, image: Image.Image, num_captions: int = 1) -> List[str]:
        try:
            if hasattr(self.model, "generate_captions"):
                return self.model.generate_captions(
                    image,
                    num_captions=num_captions,
                    max_length=MAX_LEN,
                    temperature=0.8, top_p=0.9, top_k=50,
                    use_beam_search=(num_captions == 1),
                )
            elif hasattr(self.model, "generate_diverse_captions") and num_captions > 1:
                return self.model.generate_diverse_captions(image, num_captions=num_captions)
            elif hasattr(self.model, "generate_caption"):
                cap = self.model.generate_caption(image, max_length=MAX_LEN)
                return [cap] * num_captions if cap else [""]
        except Exception as e:
            print(f"[ERROR] ViT-T5 caption failed: {e}")
        return [""] * num_captions


def load_captioners(ckpt_dir: str = "checkpoints", device: str | None = None) -> Dict[str, BaseCaptioner]:
    """
    Load toàn bộ captioners và trả dict {model_name: captioner}
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading captioners on device: {device}")

    captioners: Dict[str, BaseCaptioner] = {}
    configs = [
        ("cnn_lstm", "cnn_lstm.pth", CaptionerCNNLSTM),
        ("cnn_t5",   "cnn_t5.pt",   CaptionerCNNT5),
        ("vit_t5",   "vit_t5.pt",   CaptionerViTT5),
    ]
    for name, ckpt_file, Cls in configs:
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        try:
            cap = Cls(ckpt_path, device)
            captioners[name] = cap
            print(f"[INFO] Ready: {name}")
        except Exception as e:
            print(f"[ERROR] Load failed for {name}: {e}")

    if not captioners:
        print("[WARN] No captioners were loaded successfully!")
    return captioners
