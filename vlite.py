"""vlite: a tiny vector database. numpy is the only dependency.

    text    -> wordpiece tokens -> 6-layer BERT (mxbai-embed-xsmall-v1) -> mean over tokens -> x (384,)
    b(x)    packbits(x > 0)                                                (48,) uint8
    d(a, b) popcount(a XOR b)                                              hamming distance in [0, 384]

The tokenizer and the model are written out below in plain numpy and match the Hugging Face
implementations. The weights (48 MB) download once to ~/.cache/vlite.

The collection is four aligned columns of length N: ids, texts, metadata (1-D object arrays)
and bits (N, 48). Every row operation is one mask or one concatenate over all four.
save() writes them to a single .ctx file.
"""
import itertools
import json
import os
import re
import struct
import unicodedata
import urllib.request
import uuid

import numpy as np

MODEL = "mixedbread-ai/mxbai-embed-xsmall-v1"
CACHE = os.path.expanduser("~/.cache/vlite")
HEADS, MAX_TOKENS = 12, 512  # chunks hold MAX_TOKENS - 2 tokens so [CLS] and [SEP] fit
BITS = 384                   # one sign bit per embedding dim
BYTES = BITS // 8            # 48 bytes per vector
MAGIC, VERSION = b"CTXF", 2
CJK = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F),
       (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F)]
WORD = re.compile(r"[^\t\n\r    -     　]+")  # split on BERT's whitespace (\t\n\r + Unicode Zs/Zl/Zp), not Python's \s


# ---- weights ------------------------------------------------------------------------------

def fetch(name):
    """Path to a file from the model's Hugging Face repo, downloaded on first use."""
    path = os.path.join(CACHE, MODEL, name)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(f"https://huggingface.co/{MODEL}/resolve/main/{name}", path + ".tmp")
        os.replace(path + ".tmp", path)
    return path


def load_safetensors(path):
    """safetensors layout: u64 header length | json {name: {dtype, shape, data_offsets}} | raw bytes."""
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<Q", data[:8])[0]
    dtype = {"F16": np.float16, "F32": np.float32}
    return {k: np.frombuffer(data[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]], dtype[v["dtype"]]).reshape(v["shape"]).astype(np.float32)
            for k, v in json.loads(data[8:8 + n]).items() if k != "__metadata__"}


# ---- tokenizer: BERT uncased WordPiece ------------------------------------------------------

def is_punct(c):
    o = ord(c)
    return 33 <= o <= 47 or 58 <= o <= 64 or 91 <= o <= 96 or 123 <= o <= 126 or unicodedata.category(c).startswith("P")


def normalize(text):
    """BERT's normalizer: whitespace -> space, drop control chars, pad CJK chars with spaces, strip accents, lowercase."""
    out = []
    for c in text:
        cat = unicodedata.category(c)
        if c in "\t\n\r " or cat in ("Zs", "Zl", "Zp"):
            out.append(" ")
        elif ord(c) in (0, 0xFFFD) or cat.startswith("C"):
            continue
        elif any(lo <= ord(c) <= hi for lo, hi in CJK):
            out.append(f" {c} ")
        else:
            out.append(c)
    # accents go first (NFD splits é into e + a combining mark, then we drop the mark), then lowercase char by char
    return "".join(c.lower() for c in unicodedata.normalize("NFD", "".join(out)) if unicodedata.category(c) != "Mn")


def words(text):
    """BERT's pre-tokenizer: split on whitespace, then every punctuation char becomes its own word."""
    out = []
    for chunk in normalize(text).split():
        for punct, run in itertools.groupby(chunk, is_punct):
            out += list(run) if punct else ["".join(run)]
    return out


def wordpiece(word, vocab):
    """Greedy longest-match-first: peel the longest vocab piece off the front; later pieces carry a ## prefix."""
    if len(word) > 100:
        return ["[UNK]"]
    pieces, start = [], 0
    while start < len(word):
        end = len(word)
        while end > start and (piece := ("##" if start else "") + word[start:end]) not in vocab:
            end -= 1
        if end == start:  # no piece fits: the whole word is unknown
            return ["[UNK]"]
        pieces.append(piece)
        start = end
    return pieces


# ---- model: BERT forward pass ---------------------------------------------------------------

def layernorm(x, w, b, eps=1e-12):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps) * w + b


def gelu(x):
    """Exact GELU, 0.5 x (1 + erf(x / sqrt 2)). numpy has no erf, so use Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)."""
    z = np.abs(x) * 2 ** -0.5  # python floats keep float32; np.sqrt(2) is a float64 scalar and would upcast everything
    t = 1 / (1 + 0.3275911 * z)
    erf = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * np.exp(-z * z)
    return 0.5 * x * (1 + np.copysign(erf, x))


def bert(w, ids, mask):
    """ids, mask: (B, T) -> last hidden states (B, T, C)."""
    B, T = ids.shape
    lin = lambda x, name: x @ w[name + ".weight"].T + w[name + ".bias"]
    ln = lambda x, name: layernorm(x, w[name + ".weight"], w[name + ".bias"])
    x = w["embeddings.word_embeddings.weight"][ids] + w["embeddings.position_embeddings.weight"][:T] + w["embeddings.token_type_embeddings.weight"][0]
    x = ln(x, "embeddings.LayerNorm")                                                    # (B, T, C)
    pad = (1 - mask)[:, None, None, :] * -1e9                                            # (B, 1, 1, T) padding keys get no attention
    for i in range(sum(k.endswith("attention.self.query.weight") for k in w)):
        p = f"encoder.layer.{i}."
        q, k, v = (lin(x, p + "attention.self." + n).reshape(B, T, HEADS, -1).transpose(0, 2, 1, 3) for n in ("query", "key", "value"))  # (B, H, T, D)
        att = q @ k.transpose(0, 1, 3, 2) * q.shape[-1] ** -0.5 + pad                    # (B, H, T, T)
        att -= att.max(-1, keepdims=True)                                                # softmax, in place
        np.exp(att, out=att)
        att /= att.sum(-1, keepdims=True)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)                            # (B, T, C)
        x = ln(x + lin(y, p + "attention.output.dense"), p + "attention.output.LayerNorm")
        x = ln(x + lin(gelu(lin(x, p + "intermediate.dense")), p + "output.dense"), p + "output.LayerNorm")
    return x


# ---- search -----------------------------------------------------------------------------------

def empty():
    """Zero-row columns: (ids, texts, metadata, bits)."""
    return np.empty(0, object), np.empty(0, object), np.empty(0, object), np.zeros((0, BYTES), np.uint8)


def matches(meta, where):
    return all(meta.get(k) == v for k, v in where.items())


def hamming_topk(bits, q, k):
    """The k rows nearest q under d(bits_i, q) = popcount(bits_i XOR q), closest first.

    bits: (N, BYTES) uint8, q: (BYTES,) uint8 -> indices (k,), distances (k,)
    """
    dist = np.bitwise_count(bits ^ q).sum(axis=1, dtype=np.int32)          # (N,)
    k = min(k, len(dist))
    top = np.argpartition(dist, k - 1)[:k] if k else np.zeros(0, np.intp)   # (k,) unordered, O(N) instead of a sort
    top = top[np.argsort(dist[top], kind="stable")]                          # (k,) closest first
    return top, dist[top]


class VLite:
    def __init__(self, collection="vlite", directory="contexts"):
        self.path = os.path.join(directory, f"{collection}.ctx")
        with open(fetch("tokenizer.json"), encoding="utf-8") as f:
            self.vocab = json.load(f)["model"]["vocab"]           # {piece: id}
        self.weights = load_safetensors(fetch("model.safetensors"))
        self.ids, self.texts, self.metadata, self.bits = empty()  # one row per chunk; a long text's chunks share an id
        if os.path.exists(self.path):
            self.load()

    @property
    def cols(self):
        return self.ids, self.texts, self.metadata, self.bits

    def tokens(self, text):
        return [piece for word in words(text) for piece in wordpiece(word, self.vocab)]

    def embed(self, texts, batch_size=8):
        """b(x) for each text: (n,) -> (n, BYTES) uint8."""
        out = [np.zeros((0, BYTES), np.uint8)]
        for i in range(0, len(texts), batch_size):
            seqs = [[self.vocab["[CLS]"], *[self.vocab[p] for p in self.tokens(t)][:MAX_TOKENS - 2], self.vocab["[SEP]"]] for t in texts[i:i + batch_size]]
            T = max(map(len, seqs))
            ids = np.array([s + [self.vocab["[PAD]"]] * (T - len(s)) for s in seqs])                   # (B, T)
            mask = (np.arange(T) < np.array([len(s) for s in seqs])[:, None]).astype(np.float32)       # (B, T)
            h = bert(self.weights, ids, mask)                                                          # (B, T, C)
            x = (h * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)                              # (B, C) mean over real tokens
            out.append(np.packbits(x > 0, axis=-1))                                                    # (B, BYTES)
        return np.concatenate(out)

    def chunk(self, text, n=MAX_TOKENS - 2):
        """Split text into runs of whole words holding <= n tokens each.
        Chunks are slices of the original string, so casing and spacing survive."""
        spans = [(m.start(), m.end(), len(self.tokens(m.group()))) for m in WORD.finditer(text)]  # a token never crosses whitespace
        if sum(k for _, _, k in spans) <= n:
            return [text]
        chunks, start, end, used = [], spans[0][0], spans[0][0], 0
        for s, e, k in spans:
            if used and used + k > n:  # this word would overflow: close the chunk before it
                chunks.append(text[start:end])
                start, used = s, 0
            end, used = e, used + k
        return chunks + [text[start:end]]

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
        Layout: MAGIC | u32 version | u32 json length | json {model, ids, texts, metadata} | (N, BYTES) uint8 bits"""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        header = json.dumps({"model": MODEL, "ids": self.ids.tolist(), "texts": self.texts.tolist(), "metadata": self.metadata.tolist()}).encode()
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
        if header["model"] != MODEL:
            raise ValueError(f"{self.path} was embedded with {header['model']}, not {MODEL}; re-add its texts")
        self.ids, self.texts, self.metadata = (np.array(header[k], dtype=object) for k in ("ids", "texts", "metadata"))
        self.bits = np.frombuffer(data[12 + n:], np.uint8).reshape(-1, BYTES)

    def clear(self):
        """Empty the collection and delete its file."""
        self.ids, self.texts, self.metadata, self.bits = empty()
        if os.path.exists(self.path):
            os.remove(self.path)

    def __repr__(self):
        return f"VLite({self.path!r}, rows={self.count()})"
