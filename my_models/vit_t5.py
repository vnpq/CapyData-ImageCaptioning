# my_models/vit_t5.py
import os
from typing import Optional
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
        return self.norm(self.proj(cls))                   # (B, embed_dim)

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

    # @torch.inference_mode()
    # def generate_caption(self, image_input, max_length=25, num_beams=5):
    #     device = next(self.encoder.parameters()).device
    #     self.eval()
    #     if isinstance(image_input, str):
    #         img = Image.open(image_input).convert('RGB')
    #     elif isinstance(image_input, torch.Tensor):
    #         from torchvision import transforms as T; img = T.ToPILImage()(image_input.squeeze(0).cpu())
    #     else:
    #         img = image_input.convert('RGB')
    #     x = self.infer_transform(img).unsqueeze(0).to(device)
    #     feats = self.encoder(x); enc = self.proj(feats).unsqueeze(1)
    #     enc_out = BaseModelOutput(last_hidden_state=enc)
    #     start = self.decoder.config.decoder_start_token_id or self.tokenizer.bos_token_id
    #     dec_in = torch.tensor([[start]], device=device)
    #     ids = self.decoder.generate(
    #         decoder_input_ids=dec_in, encoder_outputs=enc_out, max_length=max_length,
    #         eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
    #         num_beams=num_beams, early_stopping=True
    #     )
    #     return self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    
    @torch.inference_mode()
    def generate_captions(
        self, image, num_captions=1, max_length=32, temperature=0.8, top_p=0.9, top_k=50
    ):
        device = next(self.parameters()).device
        self.eval()

        px = self.infer_transform(image).unsqueeze(0).to(device)
        feats = self.encoder(px)                       # (1, d_model)
        enc   = self.proj(feats).unsqueeze(1)         # (1, 1, d_model)
        enc_out = BaseModelOutput(last_hidden_state=enc)

        # seed ngẫu nhiên cho mỗi lần gọi, cô lập RNG trong context
        seed = int.from_bytes(os.urandom(8), "little")
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(seed)
            outputs = self.decoder.generate(
                encoder_outputs=enc_out,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_length,
                num_return_sequences=num_captions,
                # fix this to use pad_token_id instead of eos_token_id
                pad_token_id=self.decoder.config.eos_token_id,
                eos_token_id=self.decoder.config.eos_token_id,
                use_cache=True,
            )

        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        uniq = []
        for t in texts:
            t = t.strip()
            if t and t not in uniq:
                uniq.append(t)
        return uniq[:num_captions]

    @torch.inference_mode()
    def generate_caption(self, image: Image.Image, max_length: int = 25) -> str:
        # Giữ compat API cũ
        caps = self.generate_captions(image, num_captions=1, max_length=max_length)
        return caps[0] if caps else ""

def build_model_vit_t5(tokenizer: PreTrainedTokenizerFast, pretrained_model_name: str = "t5-base", d_model: int = 768):
    return CaptionModel(tokenizer=tokenizer, pretrained_model_name=pretrained_model_name, d_model=d_model)

# --------- Train utilities ---------
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
