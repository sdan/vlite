# vlite

a simple and blazing fast vector database

there is no database you need to set up, no server to run, and no complex configuration. just install vlite and start using it. take the CTX file with you wherever you go. its like a browser cookie but with embeddings.

![1a3e85a6-2a3f-4092-beea-8b9d69433e80](https://github.com/sdan/vlite/assets/22898443/ed21a28e-8e2a-449b-b737-4603e4f8d0bd)

## Features

- 🔥 *Fastest* vector db retrieval with binary embeddings, less than 1.1s to search 500k documents
- 🔋 Made for RAG -- with embedding generation with [mixedbread embed-large](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) baked in
- 🍪 CTX (context) file format, a novel abstraction for storing user context similar to browser cookies
- Long texts are chunked to the model's 512-token window automatically, metadata filtering built in
- **>77.95% faster than Chroma on indexing, >422% faster on retrieval, and >3.6x smaller on disk**


## Installation

```bash
pip install vlite
```

## Usage

```python
from vlite import VLite

vdb = VLite("notes")                      # loads contexts/notes.ctx if it exists
ids = vdb.add("hello world", metadata={"artist": "adele"})
vdb.add(open("attention.txt").read())     # long texts are split into chunks automatically

for id, text, metadata, distance in vdb.retrieve("how do transformers work?", top_k=3):
    print(distance, text[:80])

vdb.retrieve("hello", where={"artist": "adele"})  # filter on metadata
vdb.delete(ids)
vdb.save()                                # nothing touches disk until you call save()
```

Bring your own loaders: vlite takes strings. For PDFs, `pypdf` or `pymupdf` get you there in two lines.

## About

vlite is a vector database built for agents, ChatGPT Plugins, and other AI apps that need a fast and simple database to store vectors. It was developed to support the billions of embeddings generated, indexed, and sorted with [ChatWith+ ChatGPT Plugins](https://plugins.sdan.io/), which run for millions of users. Most vector databases either repeatedly crashed on a daily basis or were too expensive for the high throughput required.

vlite introduces the CTX file format, which acts like a browser cookie for user embeddings, providing efficient storage, retrieval of embeddings, composability, portability, and user context.

![converted copy](https://github.com/sdan/vlite/assets/22898443/1b5b330d-0094-4da1-8d01-302255aa2010)

## License

AGPL-3.0 License

## Contributing

Thanks to [Claude](https://claude.ai) and [Ray](https://github.com/raydelvecchio) for their contributions to vlite. If you'd like to contribute, please open an issue or a pull request.