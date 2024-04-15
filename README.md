# vlite

a simple and blazing fast vector database

there is no database you need to set up, no server to run, and no complex configuration. just install vlite and start using it. take the CTX file with you wherever you go. its like a browser cookie but with embeddings.

![1a3e85a6-2a3f-4092-beea-8b9d69433e80](https://github.com/sdan/vlite/assets/22898443/ed21a28e-8e2a-449b-b737-4603e4f8d0bd)

## Features

- 🔥 *Fastest* vector db retrieval with binary embeddings
- 🔋 Made for RAG -- with embedding generation baked in
- 🍪 CTX (context) file format, a novel abstraction for storing user context similar to browser cookies
- Ingest text, PDF, CSV, PPTX, and webpages
- Chunking, metadata filtering, PDF OCR support for extracting text from scanned PDFs
- **Over 77.95% faster than Chroma on indexing, and 422% faster on retrieval**


## Installation

```bash
pip install vlite
```

### Installation with PDF OCR Support

To enable PDF OCR support (with [surya](https://github.com/VikParuchuri/surya)), install the `vlite[ocr]` extra:

```bash
pip install vlite[ocr]
```

## Usage

```python
from vlite import VLite
from vlite.utils import process_pdf

vdb = VLite()
vdb.add("hello world", metadata={"artist": "adele"})
vdb.add(process_pdf("attention-is-all-you-need.pdf", use_ocr=True))

results = vdb.retrieve("how do transformers work?")
print(results)
```

## About

vlite is a vector database built for agents, ChatGPT Plugins, and other AI apps that need a fast and simple database to store vectors. It was developed to support the billions of embeddings generated, indexed, and sorted with [ChatWith+ ChatGPT Plugins](https://plugins.sdan.io/), which run for millions of users. Most vector databases either repeatedly crashed on a daily basis or were too expensive for the high throughput required.

vlite introduces the CTX file format, which acts like a browser cookie for user embeddings, providing efficient storage, retrieval of embeddings, composability, portability, and user context.

at its core is just a dictionary of embeddings, but it's optimized for speed and simplicity. There is no database you need to set up, no server to run, and no complex configuration. Just install the package and start using it. Take the CTX file with you wherever you go. It's like a browser cookie for your embeddings.

![converted copy](https://github.com/sdan/vlite/assets/22898443/1b5b330d-0094-4da1-8d01-302255aa2010)

## License

AGPL-3.0 License

## Contributing

Thanks to [Claude](https://claude.ai) and [Ray](https://github.com/raydelvecchio) for their contributions to vlite. If you'd like to contribute, please open an issue or a pull request.