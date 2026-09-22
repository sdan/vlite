# vlite architecture

vlite v1 is one file, `vlite.py` (~280 lines), and numpy is its only dependency. Read it top to bottom:

| section | lines | what it does |
|---|---|---|
| weights | ~20 | download `tokenizer.json` + `model.safetensors` once to `~/.cache/vlite`; parse safetensors with numpy |
| tokenizer | ~50 | BERT uncased WordPiece: normalize, split on whitespace/punctuation, greedy longest-match |
| model | ~35 | 6-layer BERT forward pass (mxbai-embed-xsmall-v1, 24M params) |
| search | ~140 | the collection: chunk, embed, add, retrieve, get, delete, save, load |

The v2 direction (a small RL-trained search model) lives on the `v2` branch and does not share code with v1.

## Data flow

```
add(texts)                                          retrieve(query)
  │                                                   │
  ├─ chunk()   pack whole words into ≤510 tokens,     ├─ where? → keep rows whose metadata matches (exact)
  │            slicing the original string            ├─ embed([query]) → uint8[48]
  ├─ embed()   wordpiece → bert() → mean over tokens  └─ hamming_topk()  popcount(bits ^ q), argpartition
  │            → 384 dims → sign → packbits                  → [(id, text, metadata, distance)]
  └─ append a row per chunk to the four columns
```

## Data model

The collection is four parallel columns, one row per chunk:

| column     | type                  | notes                                          |
|------------|-----------------------|------------------------------------------------|
| `ids`      | `object[N]` (str)     | one id per `add()`ed text; its chunks share it |
| `texts`    | `object[N]` (str)     | chunk text, exactly as given (not re-decoded)  |
| `metadata` | `object[N]` (dict)    | copied per row, so chunks never alias          |
| `bits`     | `np.uint8[N, 48]`     | packed sign bits, one contiguous buffer        |

All four are numpy arrays, so every row operation is one expression over `cols`:
`add` concatenates, `delete` applies `~mask`, `get`/`retrieve` index with `mask(ids, where)`.
Search scans `bits` directly. Nothing is rebuilt per query.

## `.ctx` file (v2)

```
b"CTXF" | u32 version=2 | u32 n | n bytes of JSON {model, ids, texts, metadata} | N*48 bytes of bits
```

- `save()` always writes the whole collection to `<path>.tmp`, then `os.replace` → atomic, no append mode.
- `load()` refuses other versions and other embedding models instead of returning garbage rankings.
- v1 `.ctx` files (≤0.2.x) aren't readable. They stored each 64-byte code as 64 float32s and could hold duplicated rows, so re-add instead of migrating.

## Why it looks like this

- **numpy only, verified against the reference.** The tokenizer and model are written from scratch, then checked once against Hugging Face `tokenizers` and `transformers` (neither is a dependency):
  - tokens identical on real docs, 20 unicode edge cases, and 3,000 random unicode strings. One deliberate difference: a literal `"[SEP]"` in your text stays text, while HF turns it into a control token.
  - embeddings: max abs difference 1.9e-6, 0 of 11,136 sign bits flipped, batched == one-at-a-time.
- **xsmall over large.** 48 MB of weights instead of 670 MB, and 1 dependency instead of 35 packages. On the README's Borges queries it ranked better than mxbai-embed-large did ("perfect memory" finds Funes; large picked the coin). Cost: CPU only, ~0.26 s per full 512-token chunk.
- **float32 everywhere.** Under NumPy 2, `float32_array / np.sqrt(2)` silently becomes float64, because numpy scalars are strongly typed. Scales are python floats (`2 ** -0.5`) so the forward pass stays float32, which halved its runtime.
- **Chunks end on whitespace.** BERT never lets a token cross whitespace, so per-word token counts add up exactly and chunks can be packed from whole words. `WORD` splits on BERT's whitespace set, not Python's `\s`, which also matches control chars BERT deletes.
- **Exact hamming scan, no ANN index.** ~15 ms for top-10 over 500k rows. At this scale an index adds code and approximation error for no gain.
- **Filter before search.** Post-filtering a top-k·m candidate list can silently return fewer than k results.
- **Explicit `save()`.** Mutations are in-memory only, so you always know when disk changes.
- **Strings in, tuples out.** File loaders (PDF/DOCX/CSV/web/OCR), the FastAPI server, telemetry, and the LangChain surface were removed. Each pulled in heavy dependencies or was broken.
- **No test suite.** Bugs the old code had, so any rewrite should re-check them: summed XOR instead of popcount, truncation instead of chunking, post- instead of pre-filtering, and duplicate-on-save.
