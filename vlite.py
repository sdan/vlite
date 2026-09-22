"""vlite: a tiny vector database.

    x       CLS embedding of a chunk, first 512 of 1024 dims    (512,) float
    b(x)    packbits(x > 0)                                     (64,)  uint8
    d(a, b) popcount(a XOR b)                                   hamming distance in [0, 512]

The collection is four aligned columns of length N: ids, texts, metadata (1-D object arrays)
and bits (N, 64). Every row operation is one mask or one concatenate over all four.
save() writes them to a single .ctx file.
"""
import json
import os
import struct
import uuid

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

BITS = 512         # mxbai is matryoshka-trained, so the first 512 of 1024 dims keep most of the signal
BYTES = BITS // 8  # 64 bytes per vector
MAGIC, VERSION = b"CTXF", 2


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty():
    """Zero-row columns: (ids, texts, metadata, bits)."""
    return np.empty(0, object), np.empty(0, object), np.empty(0, object), np.zeros((0, BYTES), np.uint8)


def matches(meta, where):
    return all(meta.get(k) == v for k, v in where.items())


def hamming_topk(bits, q, k):
    """The k rows nearest q under d(bits_i, q) = popcount(bits_i XOR q), closest first.

    bits: (N, 64) uint8, q: (64,) uint8 -> indices (k,), distances (k,)
    """
    dist = np.bitwise_count(bits ^ q).sum(axis=1, dtype=np.int32)          # (N,)
    k = min(k, len(dist))
    top = np.argpartition(dist, k - 1)[:k] if k else np.zeros(0, np.intp)   # (k,) unordered, O(N) instead of a sort
    top = top[np.argsort(dist[top], kind="stable")]                          # (k,) closest first
    return top, dist[top]


class VLite:
    def __init__(self, collection="vlite", device=None, model_name="mixedbread-ai/mxbai-embed-large-v1", directory="contexts"):
        self.path = os.path.join(directory, f"{collection}.ctx")
        self.device = device or default_device()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        assert self.model.config.hidden_size >= BITS, f"{model_name} has fewer than {BITS} dims"
        self.ids, self.texts, self.metadata, self.bits = empty()  # one row per chunk; a long text's chunks share an id
        if os.path.exists(self.path):
            self.load()

    @property
    def cols(self):
        return self.ids, self.texts, self.metadata, self.bits

    @torch.inference_mode()
    def embed(self, texts, batch_size=32):
        """b(x) for each text: (n,) -> (n, 64) uint8."""
        out = [np.zeros((0, BYTES), np.uint8)]
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(list(texts[i:i + batch_size]), padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            x = self.model(**batch).last_hidden_state[:, 0, :BITS]    # (B, 512) CLS token
            out.append(np.packbits((x > 0).cpu().numpy(), axis=-1))   # (B, 64) sign bits; normalizing first wouldn't change them
        return np.concatenate(out)

    def chunk(self, text, n=510):
        """Split text into spans of <= n tokens (512 minus [CLS] and [SEP]).
        Slices the original string by token offsets, so stored text keeps its casing and spacing."""
        spans = np.array(self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]).reshape(-1, 2)  # (T, 2) chars
        if len(spans) <= n:
            return [text]
        last = np.minimum(np.arange(n, len(spans) + n, n), len(spans)) - 1   # (C,) last token of each chunk
        return [text[s:e] for s, e in zip(spans[::n, 0], spans[last, 1])]

    def add(self, texts, metadata=None):
        """Add one text or a list. metadata: one dict for all texts, or a list with one dict per text. Returns ids."""
        texts = [texts] if isinstance(texts, str) else list(texts)
        metas = metadata if isinstance(metadata, list) else [metadata] * len(texts)
        ids = [uuid.uuid4().hex for _ in texts]
        rows = [(i, c, dict(m or {})) for i, t, m in zip(ids, texts, metas, strict=True) for c in self.chunk(t)]  # dict(): chunks never share one
        new = [np.array([r[j] for r in rows], dtype=object) for j in range(3)]
        self.ids, self.texts, self.metadata, self.bits = map(np.concatenate, zip(self.cols, new + [self.embed(new[1])]))
        return ids

    def mask(self, ids=None, where=None):
        """(N,) bool: rows whose id is in ids and whose metadata matches where. None means no constraint."""
        m = np.ones(len(self.ids), bool)
        if ids is not None:
            m &= np.isin(self.ids, [ids] if isinstance(ids, str) else list(ids))
        if where:
            m &= np.array([matches(x, where) for x in self.metadata], bool)
        return m

    def retrieve(self, text, top_k=5, where=None):
        """Rows nearest to text -> [(id, text, metadata, distance)], closest first.
        where filters before the search, so you get top_k results whenever that many rows match."""
        rows = np.flatnonzero(self.mask(where=where))                    # (M,)
        top, dist = hamming_topk(self.bits[rows] if where else self.bits, self.embed([text])[0], top_k)
        r = rows[top]
        return list(zip(self.ids[r], self.texts[r], self.metadata[r], dist.tolist()))

    def get(self, ids=None, where=None):
        """Rows matching ids and/or where -> [(id, text, metadata)]."""
        m = self.mask(ids, where)
        return list(zip(self.ids[m], self.texts[m], self.metadata[m]))

    def delete(self, ids):
        """Remove every row belonging to these ids. Returns the number of rows removed."""
        drop = self.mask(ids=ids)
        self.ids, self.texts, self.metadata, self.bits = (c[~drop] for c in self.cols)
        return int(drop.sum())

    def count(self):
        return len(self.ids)

    def save(self):
        """Overwrite the .ctx file with the in-memory collection.
        Layout: MAGIC | u32 version | u32 json length | json {model, ids, texts, metadata} | (N, 64) uint8 bits"""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        header = json.dumps({"model": self.model_name, "ids": self.ids.tolist(), "texts": self.texts.tolist(), "metadata": self.metadata.tolist()}).encode()
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(MAGIC + struct.pack("<II", VERSION, len(header)) + header + self.bits.tobytes())
        os.replace(tmp, self.path)  # atomic: a crash mid-write never leaves a half-written collection

    def load(self):
        with open(self.path, "rb") as f:
            data = f.read()
        version, n = struct.unpack_from("<II", data, 4)
        if data[:4] != MAGIC or version != VERSION:
            raise ValueError(f"{self.path} is not a v{VERSION} .ctx file (collections from vlite<=0.2 need re-adding)")
        header = json.loads(data[12:12 + n])
        if header["model"] != self.model_name:
            raise ValueError(f"{self.path} was embedded with {header['model']}, not {self.model_name}")
        self.ids, self.texts, self.metadata = (np.array(header[k], dtype=object) for k in ("ids", "texts", "metadata"))
        self.bits = np.frombuffer(data[12 + n:], np.uint8).reshape(-1, BYTES)

    def clear(self):
        """Empty the collection and delete its file."""
        self.ids, self.texts, self.metadata, self.bits = empty()
        if os.path.exists(self.path):
            os.remove(self.path)

    def __repr__(self):
        return f"VLite({self.path!r}, rows={self.count()}, model={self.model_name!r})"
