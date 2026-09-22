"""vlite: a tiny vector database.

text -> chunks that fit the model -> CLS embedding -> sign bit of the first 512 dims -> 64 packed bytes.
The whole collection lives in memory as four parallel columns; search is an exact hamming scan.
save() writes everything to one .ctx file.
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


def matches(meta, where):
    return all(meta.get(k) == v for k, v in where.items())


def hamming_topk(bits, q, k):
    """Exact search. bits: uint8[N, BYTES], q: uint8[BYTES] -> (row indices, distances), closest first.
    Hamming distance = number of differing bits = popcount(row XOR q)."""
    dist = np.bitwise_count(bits ^ q).sum(axis=1, dtype=np.int32)
    k = min(k, len(dist))
    top = np.argpartition(dist, k - 1)[:k] if k else np.zeros(0, np.intp)  # O(N) select, no full sort
    top = top[np.argsort(dist[top], kind="stable")]
    return top, dist[top]


class VLite:
    def __init__(self, collection="vlite", device=None, model_name="mixedbread-ai/mxbai-embed-large-v1", directory="contexts"):
        self.path = os.path.join(directory, f"{collection}.ctx")
        self.device = device or default_device()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        assert self.model.config.hidden_size >= BITS, f"{model_name} has fewer than {BITS} dims"
        # One row per chunk. A long text becomes several rows that share one id.
        self.ids, self.texts, self.metadata, self.bits = [], [], [], np.zeros((0, BYTES), np.uint8)
        if os.path.exists(self.path):
            self.load()

    @torch.inference_mode()
    def embed(self, texts, batch_size=32):
        """list of str -> uint8[len(texts), BYTES]."""
        out = [np.zeros((0, BYTES), np.uint8)]
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            cls = self.model(**batch).last_hidden_state[:, 0, :BITS]
            out.append(np.packbits((cls > 0).cpu().numpy(), axis=-1))  # sign bit only; normalizing first wouldn't change it
        return np.concatenate(out)

    def chunk(self, text, max_tokens=510):
        """Split text into pieces the model sees whole (512 minus [CLS] and [SEP]).
        Slices the original string by token offsets, so stored text keeps its casing and spacing."""
        offsets = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
        if len(offsets) <= max_tokens:
            return [text]
        return [text[offsets[i][0]:offsets[min(i + max_tokens, len(offsets)) - 1][1]] for i in range(0, len(offsets), max_tokens)]

    def add(self, texts, metadata=None):
        """Add one text or a list. metadata: one dict for all texts, or a list with one dict per text. Returns ids."""
        texts = [texts] if isinstance(texts, str) else list(texts)
        metas = metadata if isinstance(metadata, list) else [metadata] * len(texts)
        ids, rows = [], []
        for text, meta in zip(texts, metas, strict=True):
            ids.append(uuid.uuid4().hex)
            rows += [(ids[-1], chunk, dict(meta or {})) for chunk in self.chunk(text)]  # copy: chunks must not share a dict
        self.ids += [r[0] for r in rows]
        self.texts += [r[1] for r in rows]
        self.metadata += [r[2] for r in rows]
        self.bits = np.concatenate([self.bits, self.embed([r[1] for r in rows])])
        return ids

    def retrieve(self, text, top_k=5, where=None):
        """Rows nearest to text -> [(id, text, metadata, hamming distance)], closest first.
        where: exact-match metadata filter, applied before the search so you always get top_k when enough rows match."""
        rows, bits = np.arange(len(self.ids)), self.bits
        if where:
            rows = np.flatnonzero([matches(m, where) for m in self.metadata])
            bits = self.bits[rows]
        top, dist = hamming_topk(bits, self.embed([text])[0], top_k)
        return [(self.ids[r], self.texts[r], self.metadata[r], int(d)) for r, d in zip(rows[top], dist)]

    def get(self, ids=None, where=None):
        """Rows matching ids and/or a metadata filter -> [(id, text, metadata)]."""
        ids = {ids} if isinstance(ids, str) else set(ids or [])
        return [(i, t, m) for i, t, m in zip(self.ids, self.texts, self.metadata)
                if (not ids or i in ids) and (not where or matches(m, where))]

    def delete(self, ids):
        """Remove every row belonging to these ids. Returns the number of rows removed."""
        ids = {ids} if isinstance(ids, str) else set(ids)
        keep = [i not in ids for i in self.ids]
        self.ids, self.texts, self.metadata = ([x for x, k in zip(col, keep) if k] for col in (self.ids, self.texts, self.metadata))
        self.bits = self.bits[np.array(keep, dtype=bool)]
        return keep.count(False)

    def count(self):
        return len(self.ids)

    def save(self):
        """Overwrite the .ctx file with the in-memory collection.
        Layout: MAGIC | u32 version | u32 json length | json {model, ids, texts, metadata} | uint8[N, BYTES] bits"""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        header = json.dumps({"model": self.model_name, "ids": self.ids, "texts": self.texts, "metadata": self.metadata}).encode()
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
        self.ids, self.texts, self.metadata = header["ids"], header["texts"], header["metadata"]
        self.bits = np.frombuffer(data[12 + n:], np.uint8).reshape(-1, BYTES)

    def clear(self):
        """Empty the collection and delete its file."""
        self.ids, self.texts, self.metadata, self.bits = [], [], [], np.zeros((0, BYTES), np.uint8)
        if os.path.exists(self.path):
            os.remove(self.path)

    def __repr__(self):
        return f"VLite({self.path!r}, rows={self.count()}, model={self.model_name!r})"
