# vlite architecture

vlite v1 is one file, `vlite.py` (~160 lines). Read it top to bottom; there is nothing else.
The v2 direction (a small RL-trained search model) lives in `TODO.md` / `RESEARCH.md` and does not share code with v1.

## Data flow

```
add(texts)                                        retrieve(query)
  │                                                 │
  ├─ chunk()   split to ≤510 model tokens,          ├─ where? → keep rows whose metadata matches (exact)
  │            slicing the original string          ├─ embed([query]) → uint8[64]
  ├─ embed()   CLS of mxbai-embed-large             └─ hamming_topk()  popcount(bits ^ q), argpartition
  │            → first 512 dims → sign → packbits          → [(id, text, metadata, distance)]
  └─ append a row per chunk to the four columns
```

## Data model

The collection is four parallel columns, one row per chunk:

| column     | type                  | notes                                          |
|------------|-----------------------|------------------------------------------------|
| `ids`      | `object[N]` (str)     | one id per `add()`ed text; its chunks share it |
| `texts`    | `object[N]` (str)     | chunk text, exactly as given (not re-decoded)  |
| `metadata` | `object[N]` (dict)    | copied per row, so chunks never alias          |
| `bits`     | `np.uint8[N, 64]`     | packed sign bits, one contiguous buffer        |

All four are numpy arrays, so every row operation is one expression over `cols`:
`add` concatenates, `delete` applies `~mask`, `get`/`retrieve` index with `mask(ids, where)`.
Search scans `bits` directly. Nothing is rebuilt per query.

## `.ctx` file (v2)

```
b"CTXF" | u32 version=2 | u32 n | n bytes of JSON {model, ids, texts, metadata} | N*64 bytes of bits
```

- `save()` always writes the whole collection to `<path>.tmp`, then `os.replace` → atomic, no append mode.
- `load()` refuses other versions and other embedding models instead of returning garbage rankings.
- v1 `.ctx` files (≤0.2.x) aren't readable. They stored each 64-byte code as 64 float32s and could hold duplicated rows, so re-add instead of migrating.

## Why it looks like this

- **Exact hamming scan, no ANN index.** Measured ~15 ms for top-10 over 500k rows in numpy on an M-series laptop. At this scale an index adds code and approximation error for no gain.
- **Filter before search.** Post-filtering a top-k·m candidate list can silently return fewer than k results.
- **Explicit `save()`.** Mutations are in-memory only, so you always know when disk changes.
- **Strings in, tuples out.** File loaders (PDF/DOCX/CSV/web/OCR), the FastAPI server, telemetry, and the LangChain surface were removed. Each pulled in heavy dependencies or was broken, and none is core to "embed, pack, search, persist".
- **No test suite.** Bugs the old code had, so any rewrite should re-check them: summed XOR instead of popcount, truncation instead of chunking, post- instead of pre-filtering, and duplicate-on-save.
