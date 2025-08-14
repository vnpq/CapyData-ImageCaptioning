import os, json
from typing import Tuple, Dict, Optional
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import Tokenizer
from tokenizers import models as tokenizers_models
from tokenizers import pre_tokenizers
from collections import Counter

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def load_shared_tokenizer(
    vocab_json: Optional[str],
    hf_fallback: str = "t5-base",
    max_len: int = 25
) -> PreTrainedTokenizerFast:
    """
    Load tokenizer from vocab.json or Hugging Face model.
    """
    if vocab_json and os.path.exists(vocab_json):
        with open(vocab_json, 'r', encoding='utf-8') as f:
            idx2tok = json.load(f)  # { "0": "<pad>", "1": "<s>", ... }
        tok2idx = {tok: int(idx) for idx, tok in idx2tok.items()}
        wordlevel = tokenizers_models.WordLevel(vocab=tok2idx, unk_token="<unk>")
        tok = Tokenizer(wordlevel)
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_fallback)
        for t in ("<s>", "</s>", "<pad>", "<unk>"):
            if t not in tokenizer.get_vocab():
                tokenizer.add_tokens([t])
    tokenizer.model_max_length = max_len
    return tokenizer


def token_dicts_from_tokenizer(tok: PreTrainedTokenizerFast) -> Tuple[Dict[str,int], Dict[int,str]]:
    """ 
    Convert a PreTrainedTokenizerFast to token2idx and idx2token dictionaries.
    """
    token2idx = tok.get_vocab()
    idx2token = {i: t for t, i in token2idx.items()}
    return token2idx, idx2token
