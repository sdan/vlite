# vlite
A blazing fast, lightweight, and simple vector database made with numpy and llama.cpp in ~1k lines of code.

[vlite Documentation](docs.md)

![1a3e85a6-2a3f-4092-beea-8b9d69433e80](https://github.com/sdan/vlite/assets/22898443/ed21a28e-8e2a-449b-b737-4603e4f8d0bd)


## features
- 🔥 *fastest* vector db retrieval with binary embeddings and int8 rescoring 
- 🏎️ accelerated embedding generation with llama.cpp
- 🍪 OMOM (pronounced "om-nom") file format, a novel abstraction for storing user context similar to browser cookies
- injest text, PDF, CSV, PPTX, and webpages
- batteries included chunking, metadata filtering, PDF OCR support for extracting text from scanned PDFs
- **over 77.95% faster than Chroma on indexing, and 422% faster on retrieval**

## installation
```bash
pip install vlite
```

### installation with PDF OCR Support
To enable PDF OCR support (with [surya](https://github.com/VikParuchuri/surya)), install the `vlite[ocr]` extra:
```bash
pip install vlite[ocr]
```

## usage
```python
from vlite import VLite
from vlite.utils import process_pdf

vdb = VLite()

vdb.add("hello world", metadata={"artist": "adele"}

vdb.add(process_pdf("attention-is-all-you-need.pdf", use_ocr=True))

results = vdb.retrieve("how do transformers work?")

print(results)
```

## about
vlite is a vector database built for agents, ChatGPT Plugins, and other AI apps that need a fast and simple database to store vectors. It was developed to support the billions of embeddings generated, indexed, and sorted with [ChatWith+ ChatGPT Plugins](https://plugins.sdan.io/), which run for millions of users. Most vector databases either repeatedly crashed on a daily basis or were too expensive for the high throughput required.

vlite introduces the OMOM (pronounced "om-nom") file format, which acts like a browser cookie for user embeddings, providing efficient storage, retrieval of embeddings, composability, portability, and user context.

Under the hood, vlite uses llama.cpp for accelerated embedding generation and defaults to binary embeddings and INT8 embedding rescoring for the fastest retrieval in memory vector databases. It beats Chroma on all metrics retrieval/indexing, specifically 77.95% faster indexing speed compared to Chroma.
#
![converted copy](https://github.com/sdan/vlite/assets/22898443/1b5b330d-0094-4da1-8d01-302255aa2010)

## License
AGPL-3.0 License

## Contributing
Thanks to [Claude](https://claude.ai), [Ray](https://github.com/raydelvecchio), and [Howard](https://github.com/HowieG) for their contributions to vlite. If you'd like to contribute, please open an issue or a pull request.
