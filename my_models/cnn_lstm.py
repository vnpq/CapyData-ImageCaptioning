# my_models/cnn_lstm.py
import os, math, random
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from tqdm.auto import tqdm

# ---------------- Encoder CNN (ResNet-50) ----------------
class EncoderCNN(nn.Module):
    def __init__(self, encoded_size: int = 256):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # freeze all
        for p in resnet.parameters():
            p.requires_grad = False
        modules = list(resnet.children())  # [..., layer4, avgpool, fc]
        # unfreeze two last modules in original list (avgpool, fc)
        for layer in modules[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
        self.resnet = nn.Sequential(*modules[:-2])       # until layer4
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Linear(2048, encoded_size)
        self.bn  = nn.BatchNorm1d(encoded_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.resnet(images)                 # (B,2048,h,w)
        x = self.adaptive_pool(x)               # (B,2048,1,1)
        x = x.flatten(1)                        # (B,2048)
        x = self.fc(x)                          # (B,enc)
        x = self.bn(x)                          # (B,enc)
        return x

# ---------------- Decoder RNN (LSTMCell) ----------------
class DecoderRNN(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, encoder_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.fc   = nn.Linear(hidden_size, vocab_size)
        self.dropoutOthers = nn.Dropout(0.6)
        self.dropoutLSTM   = nn.Dropout(0.6)

    def forward(self, encoder_out: torch.Tensor, captions: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        B, T = captions.size()
        V = self.fc.out_features
        emb = self.embedding(captions)                      # (B,T,E)
        emb = self.dropoutLSTM(emb)
        h = torch.zeros(B, self.hidden_size, device=encoder_out.device)
        c = torch.zeros(B, self.hidden_size, device=encoder_out.device)
        outputs = torch.zeros(B, T, V, device=encoder_out.device)
        for t in range(T):
            context = encoder_out                           # (B, enc)
            inp = torch.cat([emb[:, t], context], dim=1)   # (B, E+enc)
            inp = self.dropoutLSTM(inp)
            h, c = self.lstm(inp, (h, c))
            h = self.dropoutOthers(h)
            outputs[:, t] = self.fc(h)
        return outputs

# ---------------- Kết hợp ----------------
class CaptionModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoded_size: int = 256,
        embed_size: int = 256,
        hidden_size: int = 512,
    ):
        super().__init__()
        self.encoder = EncoderCNN(encoded_size=encoded_size)
        self.decoder = DecoderRNN(embed_size=embed_size, hidden_size=hidden_size,
                                  vocab_size=vocab_size, encoder_dim=encoded_size)

        # inference transform (ImageNet norm)
        self.infer_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def forward(self, images: torch.Tensor, captions: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        feats = self.encoder(images)
        logits = self.decoder(feats, captions, teacher_forcing_ratio)
        return logits

    # @torch.inference_mode()
    # def generate_caption(self, image: torch.Tensor, vocab: Dict[str, int], idx_to_word: Dict[int, str],
    #                      max_length: int = 20, beam_width: int = 5):
    #     device = next(self.parameters()).device
    #     self.eval()
    #     if image.ndim == 3:
    #         image = image.unsqueeze(0)
    #     image = image.to(device)
    #     features = self.encoder(image)  # (1, enc)

    #     sequences = [[[], 0.0, [features, None]]]
    #     for _ in range(max_length):
    #         candidates = []
    #         for seq, score, state in sequences:
    #             if len(seq) > 0 and seq[-1] == vocab.get("</s>"):
    #                 candidates.append([seq, score, state]); continue
    #             token_id = vocab.get("<s>") if len(seq) == 0 else seq[-1]
    #             token = torch.tensor([[token_id]], device=device)
    #             embed = self.decoder.embedding(token)
    #             context = state[0]
    #             inp = torch.cat([embed.squeeze(1), context], dim=1)
    #             if state[1] is None:
    #                 h, c = self.decoder.lstm(inp)
    #             else:
    #                 h, c = self.decoder.lstm(inp, state[1])
    #             logits = self.decoder.fc(h)
    #             probs  = torch.softmax(logits, dim=-1)
    #             top_probs, top_idx = probs.topk(beam_width, dim=-1)
    #             for i in range(beam_width):
    #                 idx = top_idx[0, i].item()
    #                 sc  = float(torch.log(top_probs[0, i]))
    #                 penalty = -0.1 * seq.count(idx)  # repetition penalty
    #                 candidates.append([seq + [idx], score + sc + penalty, [features, (h, c)]])
    #         sequences = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    #     caps = []
    #     for seq, sc, _ in sequences:
    #         words = [idx_to_word.get(i, "<unk>") for i in seq if i not in (vocab.get("<s>"), vocab.get("</s>"))]
    #         caps.append(" ".join(words))
    #     return caps
    
    @torch.inference_mode()
    def generate_captions(
        self,
        image_tensor: torch.Tensor,          
        token2idx: dict,
        idx2token: dict,
        max_length: int = 32,
        num_captions: int = 1,
        top_p: float = 0.9,
        top_k: int = 50,
        temperature: float = 0.8,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
    ) -> list[str]:
        device = next(self.parameters()).device
        self.eval()
        feats = self.encoder(image_tensor.unsqueeze(0).to(device))  # (1, enc_dim)

        def _sample_top_p(probs: torch.Tensor, top_p=0.9, top_k=50):
            # probs: (V,)
            if top_k and top_k < probs.size(0):
                topk_probs, topk_idx = torch.topk(probs, top_k)
                mask = torch.zeros_like(probs); mask[topk_idx] = topk_probs
                probs = mask
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            keep = cumsum <= top_p
            if not torch.any(keep):
                keep[0] = True
            filt_idx = sorted_idx[keep]
            filt_probs = sorted_probs[keep]
            filt_probs = filt_probs / filt_probs.sum()
            draw = torch.multinomial(filt_probs, 1)
            return filt_idx[draw]

        bos_id = token2idx.get(bos_token, token2idx.get("<s>", 1))
        eos_id = token2idx.get(eos_token, token2idx.get("</s>", 2))

        results = []
        for _ in range(num_captions):
            # RNG riêng
            g = torch.Generator(device=device)
            g.manual_seed(int.from_bytes(os.urandom(4), "little"))

            h = torch.zeros(1, self.decoder.hidden_size, device=device)
            c = torch.zeros(1, self.decoder.hidden_size, device=device)
            
            token = bos_id
            out_ids = []

            for _step in range(max_length):
                inp = torch.tensor([[token]], device=device).long()  # (1,1)
                emb = self.decoder.embedding(inp).squeeze(1)         # (1,E)
                x = torch.cat([emb, feats], dim=-1)                  # (1,E+enc)
                h, c = self.decoder.lstm(x, (h, c))                  # (1,H)
                logits = self.decoder.fc(h) / max(temperature, 1e-5) # (1,V)
                probs = F.softmax(logits.squeeze(0), dim=-1)         # (V,)
                next_token = int(_sample_top_p(probs, top_p=top_p, top_k=top_k))
                if next_token == eos_id:
                    break
                out_ids.append(next_token)
                token = next_token

            # decode
            words = []
            for tid in out_ids:
                tok = idx2token.get(int(tid), "")
                if tok and tok not in ("<pad>", "<s>", "</s>", "<unk>"):
                    words.append(tok)
            text = " ".join(words).strip()
            if text and text not in results:
                results.append(text)

        return results[:num_captions]

    @torch.inference_mode()
    def generate_caption(self, image_tensor, token2idx, idx2token, max_length=32, **kwargs):
        # Giữ API cũ – lấy caption đầu tiên từ sampling
        caps = self.generate_captions(image_tensor, token2idx, idx2token, max_length=max_length, num_captions=1)
        return caps

# --------- Dataset & Train utilities (để sẵn khi cần train) ---------
class Dataset(Dataset):
    """
    Expect captions_file lines:  'image_name.jpg<TAB>caption in Vietnamese'
    """
    def __init__(self, image_dir: str, captions_file: str, tokenizer, max_len: int = 32):
        self.image_dir = image_dir
        self.max_len = max_len
        self.tokenizer = tokenizer
        with open(captions_file, 'r', encoding='utf-8') as f:
            self.pairs = [l.strip().split('\t') for l in f if l.strip()]
        self.tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_name, caption = self.pairs[idx]
        img = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')
        img = self.tf(img)
        enc = self.tokenizer(caption, padding='max_length', truncation=True,
                             max_length=self.max_len, return_tensors='pt')
        ids = enc.input_ids.squeeze(0).long()
        return img, ids, caption

def split_loaders(dataset: Dataset, batch_size=32, val_ratio=0.1, num_workers=4, seed=42):
    n = len(dataset); n_val = int(n * val_ratio)
    n_train = n - n_val
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def train_cnn_lstm(
    model: CaptionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
    ckpt_path: str = "checkpoints/cnn_lstm.pth",
    device: Optional[str] = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    pad_id = tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best = float('inf'); es = 0
    for ep in range(1, epochs+1):
        tf_ratio = max(0.5 - 0.05*(ep-1), 0.0)
        model.train(); tr_loss = 0.0
        for images, ids, _ in tqdm(train_loader, desc=f"[CNN-LSTM] Train {ep}/{epochs}"):
            images = images.to(device)
            ids    = ids.to(device)
            optimizer.zero_grad()
            logits = model(images, ids[:, :-1], teacher_forcing_ratio=tf_ratio)  # predict next token
            loss = criterion(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
            loss.backward(); optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for images, ids, _ in val_loader:
                images = images.to(device); ids = ids.to(device)
                logits = model(images, ids[:, :-1], teacher_forcing_ratio=0.0)
                va_loss += criterion(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)).item()
        va_loss /= max(1, len(val_loader))
        scheduler.step(va_loss)

        print(f"[CNN-LSTM] Epoch {ep}: train {tr_loss:.4f}  val {va_loss:.4f}")
        if va_loss < best:
            best = va_loss; es = 0
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✔ saved best to {ckpt_path}")
        else:
            es += 1
            if es >= patience:
                print("  ⏹ early stop")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device).eval()
    return model

def build_model_cnn_lstm(vocab_size: int, encoded_size=256, embed_size=256, hidden_size=512) -> CaptionModel:
    return CaptionModel(vocab_size=vocab_size, encoded_size=encoded_size, embed_size=embed_size, hidden_size=hidden_size)
