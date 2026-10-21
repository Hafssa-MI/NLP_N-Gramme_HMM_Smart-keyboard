import random
import re
import time
from collections import Counter

import config

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
HTML_RE = re.compile(r"<[^<>]{0,40}>")          # bounded, so it can never backtrack badly


def normalize(text):
    text = HTML_RE.sub(" ", text)               # html such as <br />
    return text.lower().replace("\u2019", "'")


def tokenize(sentence):
    return TOKEN_RE.findall(normalize(sentence))


def split_sentences(text):
    return [s for s in SPLIT_RE.split(normalize(text)) if s and s.strip()]


def load_texts():
    cfg = config.DATASET
    t0 = time.time()
    if cfg["source"] == "csv":
        import pandas as pd
        texts = pd.read_csv(cfg["path"])[cfg["text_col"]].dropna().astype(str).tolist()
        random.Random(config.SEED).shuffle(texts)
        texts = texts[: config.MAX_DOCS]
    else:
        from datasets import load_dataset
        print("  loading dataset (first run downloads about 80 MB)...", flush=True)
        ds = load_dataset(cfg["name"], split=cfg["split"])
        print(f"  dataset ready: {len(ds):,} rows in {time.time() - t0:.1f}s", flush=True)
        ds = ds.shuffle(seed=config.SEED).select(range(min(config.MAX_DOCS, len(ds))))
        texts = ds[cfg["text_col"]]
        texts = [str(t) for t in texts]
    print(f"  {len(texts):,} documents in memory ({time.time() - t0:.1f}s)", flush=True)
    return texts


def build_splits(train_frac=0.8, val_frac=0.1):
    """Split BY DOCUMENT (80/10/10) so sentences of one document never straddle two splits."""
    t0 = time.time()
    docs = []
    for text in load_texts():
        sents = [tokenize(s) for s in split_sentences(text)]
        sents = [s for s in sents if 3 <= len(s) <= 60]
        if sents:
            docs.append(sents)
    print(f"  tokenized {len(docs):,} documents ({time.time() - t0:.1f}s total)", flush=True)
    a, b = int(len(docs) * train_frac), int(len(docs) * (train_frac + val_frac))
    flat = lambda ds: [s for d in ds for s in d]
    return flat(docs[:a]), flat(docs[a:b]), flat(docs[b:])


def build_vocab(train):
    counts = Counter(w for s in train for w in s)
    words = [w for w, c in counts.most_common(config.MAX_VOCAB) if c >= config.MIN_WORD_COUNT]
    return words, counts
