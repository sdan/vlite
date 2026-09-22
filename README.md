# vlite

a simple and blazing fast vector database

there is no database you need to set up, no server to run, and no complex configuration. just install vlite and start using it. take the CTX file with you wherever you go. its like a browser cookie but with embeddings.

![1a3e85a6-2a3f-4092-beea-8b9d69433e80](https://github.com/sdan/vlite/assets/22898443/ed21a28e-8e2a-449b-b737-4603e4f8d0bd)

## Features

- 🔥 *Fastest* vector db retrieval with binary embeddings, less than 1.1s to search 500k documents
- 🔋 Made for RAG -- embeddings from [mixedbread embed-xsmall](https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1) baked in, run in plain numpy. numpy is the only dependency
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

db = VLite("borges")
db.add(["a library of hexagonal rooms that holds every possible book",
        "a man who remembers every leaf of every tree he has ever seen",
        "a garden where time forks into every possible future"])

for _, text, _, distance in db.retrieve("parallel timelines", top_k=2):
    print(distance, text)

db.save()
```

```
148 a garden where time forks into every possible future
167 a man who remembers every leaf of every tree he has ever seen
```

each line is a borges story. each result is `(id, text, metadata, distance)`, and lower distance means closer. `save()` writes everything to `contexts/borges.ctx`, and `VLite("borges")` picks it back up next time. the first run downloads the model (48 MB) to `~/.cache/vlite`.

tag things with metadata, then filter on it:

```python
db.add(["a point in a cellar that contains every other point in the universe",
        "a coin that, once seen, can never be put out of mind"],
       metadata={"book": "the aleph"})

db.retrieve("perfect memory", top_k=1)                               # the man who remembers everything
db.retrieve("perfect memory", top_k=1, where={"book": "the aleph"})  # the coin you can't forget
```

long texts get chunked for you. vlite only takes strings, so bring your own pdf reader.

## About

vlite is a vector database built for agents, ChatGPT Plugins, and other AI apps that need a fast and simple database to store vectors. It was developed to support the billions of embeddings generated, indexed, and sorted with [ChatWith+ ChatGPT Plugins](https://plugins.sdan.io/), which run for millions of users. Most vector databases either repeatedly crashed on a daily basis or were too expensive for the high throughput required.

vlite introduces the CTX file format, which acts like a browser cookie for user embeddings, providing efficient storage, retrieval of embeddings, composability, portability, and user context.

![converted copy](https://github.com/sdan/vlite/assets/22898443/1b5b330d-0094-4da1-8d01-302255aa2010)

## License

AGPL-3.0 License

## Contributing

Thanks to [Claude](https://claude.ai) and [Ray](https://github.com/raydelvecchio) for their contributions to vlite. If you'd like to contribute, please open an issue or a pull request.