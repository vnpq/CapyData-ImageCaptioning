import os
from typing import Dict, List
from PIL import Image
import torch
import torchvision.transforms as T

from tokenizer import load_shared_tokenizer, token_dicts_from_tokenizer

from my_models.cnn_lstm import build_model_cnn_lstm
from my_models.cnn_t5   import build_model_cnn_t5
from my_models.vit_t5   import build_model_vit_t5

# Configuration
VOCAB_JSON  = os.getenv("VOCAB_JSON", "data/capydata_ic/vocab/idx_to_word.json")
T5_PRETRAIN = os.getenv("T5_PRETRAIN", "t5-base")
MAX_LEN     = int(os.getenv("MAX_LEN", "25"))

# Load tokenizer and vocabulary
TOKENIZER = load_shared_tokenizer(VOCAB_JSON, hf_fallback=T5_PRETRAIN, max_len=MAX_LEN)
TOKEN2IDX, IDX2TOKEN = token_dicts_from_tokenizer(TOKENIZER)
VOCAB_SIZE = len(TOKEN2IDX)

def _pp_norm224():
    """Standard ImageNet normalization for 224x224 images"""
    return T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class BaseCaptioner(torch.nn.Module):
    """Base class for all captioning models"""
    def __init__(self): 
        super().__init__()
    
    @torch.inference_mode()
    def caption(self, image: Image.Image, num_captions: int = 1) -> List[str]: 
        raise NotImplementedError
    
    def to(self, device):
        """Ensure proper device movement"""
        super().to(device)
        self.device = device
        return self

class CaptionerCNNLSTM(BaseCaptioner):
    def __init__(self, ckpt_path: str, device: str):
        super().__init__()
        self.device = device
        self.model = build_model_cnn_lstm(vocab_size=VOCAB_SIZE)
        
        # Load checkpoint
        if os.path.exists(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                # Handle different checkpoint formats
                if isinstance(state, dict) and "model" in state:
                    self.model.load_state_dict(state["model"], strict=False)
                else:
                    self.model.load_state_dict(state, strict=False)
                print(f"[INFO] Loaded CNN-LSTM checkpoint from {ckpt_path}")
            except Exception as e:
                print(f"[WARN] Failed to load CNN-LSTM checkpoint: {e}")
        else:
            print(f"[WARN] CNN-LSTM checkpoint not found: {ckpt_path}")
            
        self.model = self.model.to(device).eval()
        self.pp = _pp_norm224()
        
    @torch.inference_mode()
    def caption(self, image: Image.Image, num_captions: int = 1) -> List[str]:
        # Preprocess image
        x = self.pp(image).to(self.device)
        
        # Use new generate_captions method if available
        if hasattr(self.model, "generate_captions"):
            try:
                caps = self.model.generate_captions(
                    x, TOKEN2IDX, IDX2TOKEN,
                    max_length=MAX_LEN, 
                    num_captions=num_captions,
                    top_p=0.9, 
                    top_k=50, 
                    temperature=0.8
                )
                return caps[:num_captions] if caps else [""]
            except Exception as e:
                print(f"[WARN] generate_captions failed: {e}")
        
        # Fallback to old method
        if hasattr(self.model, "generate_caption"):
            try:
                result = self.model.generate_caption(x, TOKEN2IDX, IDX2TOKEN, max_length=MAX_LEN)
                if isinstance(result, list):
                    caption = result[0] if result else ""
                else:
                    caption = str(result)
                return [caption] * num_captions if caption else [""]
            except Exception as e:
                print(f"[WARN] generate_caption failed: {e}")
        
        return [""] * num_captions

class CaptionerCNNT5(BaseCaptioner):
    def __init__(self, ckpt_path: str, device: str):
        super().__init__()
        self.device = device
        self.model = build_model_cnn_t5(tokenizer=TOKENIZER, pretrained_model_name=T5_PRETRAIN)
        
        # Load checkpoint
        if os.path.exists(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    self.model.load_state_dict(state["model"], strict=False)
                else:
                    self.model.load_state_dict(state, strict=False)
                print(f"[INFO] Loaded CNN-T5 checkpoint from {ckpt_path}")
            except Exception as e:
                print(f"[WARN] Failed to load CNN-T5 checkpoint: {e}")
        else:
            print(f"[WARN] CNN-T5 checkpoint not found: {ckpt_path}")
            
        self.model = self.model.to(device).eval()
        
    @torch.inference_mode()
    def caption(self, image: Image.Image, num_captions: int = 1) -> List[str]:
        try:
            # Use generate_captions if available
            if hasattr(self.model, "generate_captions"):
                return self.model.generate_captions(
                    image, 
                    num_captions=num_captions, 
                    max_length=MAX_LEN,
                    repetition_penalty=1.2,
                    num_beams=3
                )
            
            # Fallback to single caption generation
            elif hasattr(self.model, "generate_caption"):
                caption = self.model.generate_caption(image, max_length=MAX_LEN)
                return [caption] * num_captions if caption else [""]
                
        except Exception as e:
            print(f"[ERROR] CNN-T5 caption generation failed: {e}")
            
        return [""] * num_captions

class CaptionerViTT5(BaseCaptioner):
    def __init__(self, ckpt_path: str, device: str):
        super().__init__()
        self.device = device
        self.model = build_model_vit_t5(
            tokenizer=TOKENIZER, 
            pretrained_model_name=T5_PRETRAIN, 
            d_model=768
        )
        
        # Load checkpoint
        if os.path.exists(ckpt_path):
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    self.model.load_state_dict(state["model"], strict=False)
                else:
                    self.model.load_state_dict(state, strict=False)
                print(f"[INFO] Loaded ViT-T5 checkpoint from {ckpt_path}")
            except Exception as e:
                print(f"[WARN] Failed to load ViT-T5 checkpoint: {e}")
        else:
            print(f"[WARN] ViT-T5 checkpoint not found: {ckpt_path}")
            
        self.model = self.model.to(device).eval()
        
    @torch.inference_mode()
    def caption(self, image: Image.Image, num_captions: int = 1) -> List[str]:
        try:
            # Use enhanced generate_captions if available
            if hasattr(self.model, "generate_captions"):
                return self.model.generate_captions(
                    image, 
                    num_captions=num_captions, 
                    max_length=MAX_LEN,
                    temperature=0.8, 
                    top_p=0.9, 
                    top_k=50,
                    use_beam_search=(num_captions == 1)  # Use beam search for single caption
                )
            
            # Use diverse captions method if available and requesting multiple
            elif hasattr(self.model, "generate_diverse_captions") and num_captions > 1:
                return self.model.generate_diverse_captions(image, num_captions=num_captions)
            
            # Fallback to single caption
            elif hasattr(self.model, "generate_caption"):
                caption = self.model.generate_caption(image, max_length=MAX_LEN)
                return [caption] * num_captions if caption else [""]
                
        except Exception as e:
            print(f"[ERROR] ViT-T5 caption generation failed: {e}")
            
        return [""] * num_captions

def load_captioners(ckpt_dir: str = "checkpoints", device: str = None) -> Dict[str, BaseCaptioner]:
    """
    Load all available captioning models
    
    Args:
        ckpt_dir: Directory containing model checkpoints
        device: Device to load models on (auto-detect if None)
        
    Returns:
        Dictionary of model_name -> captioner instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[INFO] Loading captioners on device: {device}")
    
    captioners = {}
    
    # Define model configurations
    model_configs = [
        ("cnn_lstm", "cnn_lstm.pth", CaptionerCNNLSTM),
        ("cnn_t5", "cnn_t5.pt", CaptionerCNNT5),
        ("vit_t5", "vit_t5.pt", CaptionerViTT5),
    ]
    
    for model_name, checkpoint_file, captioner_class in model_configs:
        ckpt_path = os.path.join(ckpt_dir, checkpoint_file)
        try:
            captioner = captioner_class(ckpt_path, device)
            captioners[model_name] = captioner
            print(f"[INFO] Successfully loaded {model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to load {model_name}: {e}")
    
    if not captioners:
        print("[WARN] No captioners were loaded successfully!")
    
    return captioners

def test_captioners(captioners: Dict[str, BaseCaptioner], test_image_path: str = None):
    """
    Test all loaded captioners with a sample image
    
    Args:
        captioners: Dictionary of captioners from load_captioners()
        test_image_path: Path to test image (will create dummy if None)
    """
    if test_image_path and os.path.exists(test_image_path):
        test_image = Image.open(test_image_path).convert("RGB")
    else:
        # Create a dummy test image
        import numpy as np
        test_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_array)
        print("[INFO] Using dummy test image")
    
    print(f"\n[INFO] Testing {len(captioners)} captioners...")
    
    for model_name, captioner in captioners.items():
        try:
            print(f"\n--- Testing {model_name} ---")
            
            # Test single caption
            single_caption = captioner.caption(test_image, num_captions=1)
            print(f"Single caption: {single_caption}")
            
            # Test multiple captions
            multi_captions = captioner.caption(test_image, num_captions=3)
            print(f"Multiple captions: {multi_captions}")
            
        except Exception as e:
            print(f"[ERROR] {model_name} test failed: {e}")
