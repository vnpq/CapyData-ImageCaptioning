# my_models/cnn_t5.py
import os
from typing import Optional
from PIL import Image

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from tqdm.auto import tqdm

from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast
from transformers.modeling_outputs import BaseModelOutput

class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for p in resnet.parameters(): p.requires_grad = False
        for layer in ['layer3','layer4']:
            for p in getattr(resnet, layer).parameters(): p.requires_grad = True
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])   # (B,2048,1,1)
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        feats = self.backbone(x).squeeze(-1).squeeze(-1)
        return self.norm(self.fc(feats))                                # (B,embed_dim)

class CaptionModel(nn.Module):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, pretrained_model_name="t5-base", d_model=768):
        super().__init__()
        self.encoder = CNNEncoder(embed_dim=d_model)
        self.proj    = nn.Linear(d_model, d_model)
        self.decoder = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
        self.decoder.resize_token_embeddings(len(tokenizer.get_vocab()))
        self.tokenizer = tokenizer
        # NOTE: theo notebook CNN-T5: Không normalize khi infer/train
        self.infer_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        print("[CNN-T5]", "pad:", self.tokenizer.pad_token_id, "eos:", self.tokenizer.eos_token_id,
       "start:", self.decoder.config.decoder_start_token_id)

    def forward(self, images, input_ids, attention_mask, labels=None):
        feats = self.encoder(images)
        enc   = self.proj(feats).unsqueeze(1)
        return self.decoder(encoder_outputs=(enc,), input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels, return_dict=True)

    # @torch.inference_mode()
    # def generate_caption(self, image_input, max_length=32, num_beams=5):
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
        self,
        image_input,
        num_captions: int = 1,
        max_length: int = 32,
        temperature: float = 0.9,
        top_p: float = 0.9,
        top_k: int = 50,
    ):
        device = next(self.encoder.parameters()).device
        self.eval()

        # --- preprocess ảnh ---
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, torch.Tensor):
            from torchvision import transforms as T
            img = T.ToPILImage()(image_input.squeeze(0).cpu())
        else:
            img = image_input.convert("RGB")
        x = self.infer_transform(img).unsqueeze(0).to(device)  # (1,C,H,W)

        # --- encode ảnh ---
        feats = self.encoder(x)                  # (1,d_model)
        enc   = self.proj(feats).unsqueeze(1)    # (1,1,d_model)

        # --- expand theo số caption ---
        batch_n = max(1, int(num_captions))
        if batch_n > 1:
            enc = enc.repeat(batch_n, 1, 1)      # (N,1,d_model)

        enc_out = BaseModelOutput(last_hidden_state=enc)

        # --- khởi tạo decoder input & mask ---
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.decoder.config.eos_token_id
        start_id = self.decoder.config.decoder_start_token_id or pad_id or 0

        decoder_input_ids = torch.full((batch_n, 1), start_id, dtype=torch.long, device=device)
        decoder_attention_mask = torch.ones_like(decoder_input_ids)

        # --- cấm sinh <pad> (an toàn vì pad != eos trong case của bạn) ---
        bad_words_ids = [[int(pad_id)]] if pad_id is not None and (eos_id is None or pad_id != eos_id) else None

        # --- RNG cục bộ để lần submit nào cũng khác ---
        seed = int.from_bytes(os.urandom(8), "little")
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            outputs = self.decoder.generate(
                encoder_outputs=enc_out,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_length,
                num_return_sequences=1,           # đã expand batch
                pad_token_id=pad_id,              # dùng pad thật, KHÔNG dùng eos
                eos_token_id=eos_id,              # eos=3 theo log của bạn
                bad_words_ids=bad_words_ids,
                use_cache=True,
                no_repeat_ngram_size=2,           # chống lặp n-gram đơn giản
            )

   

        texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # (chưa dedup để bạn thấy đủ n dòng; nếu muốn, có thể lọc trùng sau)
        texts = [t.strip() for t in texts if t and t.strip()]
        return texts[:num_captions] if texts else ["" for _ in range(num_captions)]

def build_model_cnn_t5(tokenizer: PreTrainedTokenizerFast, pretrained_model_name="t5-base", d_model=768):
    return CaptionModel(tokenizer=tokenizer, pretrained_model_name=pretrained_model_name, d_model=d_model)

# --------- Train utilities ---------
class Dataset(Dataset):
    """ captions_file: lines 'image.jpg<TAB>caption'  """
    def __init__(self, image_dir: str, captions_file: str, tokenizer: PreTrainedTokenizerFast, max_len: int = 32):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_len   = max_len
        with open(captions_file, 'r', encoding='utf-8') as f:
            self.pairs = [l.strip().split('\t') for l in f if l.strip()]
        self.tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])  # no normalize

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

def train_cnn_t5(
    model: CaptionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    patience: int = 3,
    ckpt_path: str = "checkpoints/cnn_t5.pt",
    device: Optional[str] = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    best = float('inf'); es = 0
    for ep in range(1, epochs+1):
        model.train(); tr = 0.0
        for imgs, ids, masks, _ in tqdm(train_loader, desc=f"[CNN-T5] Train {ep}/{epochs}"):
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

        print(f"[CNN-T5] Epoch {ep}: train {tr:.4f}  val {va:.4f}")
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
