import os
from typing import Optional, List
from PIL import Image

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm.auto import tqdm

from transformers import ViTModel, ViTConfig, T5ForConditionalGeneration, PreTrainedTokenizerFast
from transformers.modeling_outputs import BaseModelOutput

# ---------------- ViTEncoder ----------------
class ViTEncoder(nn.Module):
    def __init__(self, embed_dim: int = 768, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        else:
            cfg = ViTConfig(hidden_size=embed_dim, num_hidden_layers=12, num_attention_heads=12,
                            patch_size=16, image_size=224)
            self.vit = ViTModel(cfg)
        for name, p in self.vit.named_parameters():
            if 'encoder.layer.11' not in name:
                p.requires_grad = False
        self.proj = nn.Linear(self.vit.config.hidden_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        out = self.vit(pixel_values=x)
        cls = out.last_hidden_state[:, 0]
        return self.norm(self.proj(cls))

# ---------------- CaptionModel ----------------
class CaptionModel(nn.Module):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, pretrained_model_name: str = "t5-base", d_model: int = 768):
        super().__init__()
        self.encoder = ViTEncoder(embed_dim=d_model, pretrained=True)
        self.proj    = nn.Linear(d_model, d_model)
        self.decoder = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
        self.decoder.resize_token_embeddings(len(tokenizer.get_vocab()))
        self.tokenizer = tokenizer
        self.infer_transform = transforms.Compose([
            transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def forward(self, images, input_ids, attention_mask, labels=None):
        feats = self.encoder(images)
        enc   = self.proj(feats).unsqueeze(1)
        return self.decoder(encoder_outputs=(enc,), input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels, return_dict=True)

    @torch.inference_mode()
    def generate_captions(
        self, 
        image, 
        num_captions=1, 
        max_length=40, 
        min_length=10,
        temperature=0.7, 
        top_p=0.95, 
        top_k=40,
        repetition_penalty=1.2,
        length_penalty=1.0,
        use_beam_search=False,
        num_beams=3
    ):
        """
        Cải thiện hàm generate captions với nhiều tùy chọn hơn
        
        Args:
            image: PIL Image
            num_captions: số caption cần tạo
            max_length: độ dài tối đa
            min_length: độ dài tối thiểu
            temperature: độ "sáng tạo" (thấp = conservative, cao = creative)
            top_p: nucleus sampling threshold
            top_k: top-k sampling
            repetition_penalty: phạt lặp từ
            length_penalty: khuyến khích câu dài hơn
            use_beam_search: dùng beam search thay vì sampling
            num_beams: số beam cho beam search
        """
        device = next(self.parameters()).device
        self.eval()

        # Tiền xử lý ảnh
        px = self.infer_transform(image).unsqueeze(0).to(device)
        feats = self.encoder(px)
        enc = self.proj(feats).unsqueeze(1)
        enc_out = BaseModelOutput(last_hidden_state=enc)

        # Cấu hình generation
        generation_kwargs = {
            'encoder_outputs': enc_out,
            'max_new_tokens': max_length,
            'min_new_tokens': min_length,
            'num_return_sequences': num_captions,
            'pad_token_id': self.decoder.config.eos_token_id,
            'eos_token_id': self.decoder.config.eos_token_id,
            'use_cache': True,
            'repetition_penalty': repetition_penalty,
            'length_penalty': length_penalty,
        }
        
        if use_beam_search:
            # Beam search cho kết quả ổn định hơn
            generation_kwargs.update({
                'do_sample': False,
                'num_beams': num_beams,
                'early_stopping': True,
            })
        else:
            # Sampling cho kết quả đa dạng hơn
            generation_kwargs.update({
                'do_sample': True,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
            })

        # Generate với seed ngẫu nhiên
        seed = int.from_bytes(os.urandom(8), "little")
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(seed)
            outputs = self.decoder.generate(**generation_kwargs)

        # Decode và post-process
        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Cải thiện post-processing
        processed_captions = []
        for text in texts:
            caption = self._post_process_caption(text.strip())
            if caption and len(caption) >= 3 and caption not in processed_captions:
                processed_captions.append(caption)
        
        return processed_captions[:num_captions]

    def _post_process_caption(self, text: str) -> str:
        """
        Cải thiện chất lượng caption sau khi generate
        """
        if not text:
            return ""
        
        # Loại bỏ khoảng trắng thừa
        text = ' '.join(text.split())
        
        # Đảm bảo câu bắt đầu bằng chữ hoa
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Thêm dấu chấm nếu chưa có dấu câu ở cuối
        if text and text[-1] not in '.!?':
            text += '.'

        return text.strip()

    @torch.inference_mode() 
    def generate_caption(self, image: Image.Image, max_length: int = 25) -> str:
        """
        Hàm tương thích ngược, sử dụng beam search cho kết quả ổn định
        """
        caps = self.generate_captions(
            image, 
            num_captions=1, 
            max_length=max_length,
            use_beam_search=True,
            num_beams=5,
            repetition_penalty=1.1
        )
        return caps[0] if caps else ""

def build_model_vit_t5(tokenizer: PreTrainedTokenizerFast, pretrained_model_name: str = "t5-base", d_model: int = 768):
    return CaptionModel(tokenizer=tokenizer, pretrained_model_name=pretrained_model_name, d_model=d_model)

# --------- Train utilities (unchanged) ---------
class Dataset(Dataset):
    """ captions_file: 'image.jpg<TAB>caption' """
    def __init__(self, image_dir: str, captions_file: str, tokenizer: PreTrainedTokenizerFast, max_len: int = 32):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_len   = max_len
        with open(captions_file, 'r', encoding='utf-8') as f:
            self.pairs = [l.strip().split('\t') for l in f if l.strip()]
        self.tf = transforms.Compose([
            transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        name, cap = self.pairs[idx]
        img = Image.open(os.path.join(self.image_dir, name)).convert('RGB')
        img = self.tf(img)
        enc = self.tokenizer(cap, padding='max_length', truncation=True,
                             max_length=self.max_len, return_tensors='pt')
        ids  = enc.input_ids.squeeze(0).long()
        mask = enc.attention_mask.squeeze(0).long()
        return img, ids, mask, cap

def split_loaders(dataset: Dataset, batch_size=16, val_ratio=0.1, num_workers=4, seed=42):
    n = len(dataset); n_val = int(n * val_ratio); n_train = n - n_val
    g = torch.Generator().manual_seed(seed)
    train, val = random_split(dataset, [n_train, n_val], generator=g)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )

def train_vit_t5(
    model: CaptionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    patience: int = 3,
    ckpt_path: str = "checkpoints/vit_t5.pt",
    device: Optional[str] = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    best = float('inf'); es = 0
    for ep in range(1, epochs+1):
        model.train(); tr = 0.0
        for imgs, ids, masks, _ in tqdm(train_loader, desc=f"[ViT-T5] Train {ep}/{epochs}"):
            imgs, ids, masks = imgs.to(device), ids.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs, ids, masks, labels=ids)
            loss = out.loss
            loss.backward(); optimizer.step()
            tr += loss.item()
        tr /= max(1, len(train_loader))

        model.eval(); va = 0.0
        with torch.no_grad():
            for imgs, ids, masks, _ in val_loader:
                imgs, ids, masks = imgs.to(device), ids.to(device), masks.to(device)
                va += model(imgs, ids, masks, labels=ids).loss.item()
        va /= max(1, len(val_loader))
        scheduler.step(va)

        print(f"[ViT-T5] Epoch {ep}: train {tr:.4f}  val {va:.4f}")
        if va < best:
            best = va; es = 0
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✔ saved best to {ckpt_path}")
        else:
            es += 1
            if es >= patience:
                print("  ⏹ early stop"); break

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.to(device).eval()