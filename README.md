# VLite V2 (VLite2)

V2 is a collection and implementation of improvements upon the original amazing [VLite](https://github.com/sdan/vlite). Blazing fast and with even more features.
All written in a few hundred lines of Python! Download via `pip install vlite2`. PyPi project found [here](https://pypi.org/project/vlite2/). VLite2 is the
***fastest*** vector database you can use to start your proejct. You can find benchmarking
code comparing VLite2 to other vector databases at this repo [here](https://github.com/raydelvecchio/vdb-benchmark).

# Usage Example
```python
from vlite2 import VLite2

vdb = VLite2(vdb_name = "Example")

texts = ["Hello there", "Obi Wan Kenobi", "I've been expecting you"]

for text in texts:
  vdb.ingest(text)

results = vdb.retrieve("star wars")
```

# VLite V2 New Features

I had been using this for a few projects and found I had some more use cases not in the original. New features are
as follows:

* Allows you to retrieve metadata associated with topk texts, allowing it to be used as a semantic 
search key-value database
* We now have proper documentation on how to use below!
* All important internal variables are mangled for enhanced safety and security; only device is left exposed since that can be changed on use case
* Significantly less dependencies, and thus reduced install time
* Can now set max sequence length for chunking for each independent memorize operation; great for more varied data
across a given database
* Important input types are now enforced
* Can now define your own embedding model with a string in the `sentence_transformers` library to use internally
* When chunking, removed default newline split return; the db will now chunk ANY text, even if it already contains
some newlines (\n) within it
* Added **Ingestors** class in `vlite2/ingestors`!
  * Allows you to process strings, dicts, .txt, .pdf, and .docx files into your VLite2 database!
  * import with `from vlite2 import Ingestor` to get started
* **Naive implementation of Weaviate's [autocut](https://weaviate.io/developers/weaviate/search/hybrid#limiting-results-with-autocut-and-auto_limit)!** In the 
`retrieve` function, we no longer need to set K, rather, clusters of results are returned with each other.
  * I implement this by first finding the differences in similarity between the top X (hyperparameter) scores, calculating
  the standard deviation, then clustering by differences LARGER than the standard deviation. Results are returned
  then based on the number of CLUSTERS you want to reference, rather than direct elements.
  * If K number of relevant clusters do NOT exist, then the maximum number of clusters possible is returned by default
  * EX: `[10, 9, 4, 3, -2]` -> `[10, 9], [4, 3], [-2]`, if `k=2` we return elements associated with differences `[10, 9, 4, 3]`
* Integration with [USearch](https://github.com/unum-cloud/usearch) as the vector search engine instead of custom numpy comparisons
  * USearch handles search, vector computation, and indexing natively as the fastest vector search engine in the world!
  * Documentation for using USearch found [here](https://unum-cloud.github.io/usearch/python/reference.html)!
* Exposed methods to retrieve texts, metadata, vectors, metadata / index filenames, calculate length, and cast to string

# VLite2 Documentation

## Class `VLite2`
Main class for using the VLite2 database. Located in [main.py](vlite2/main.py). Imported via `from vlite2 import VLite2`.

* `__init__`: constructor for VLite2 class. Returns nothing.
  * **vdb_name (str)**: name of the vector database, will become the filename
  * **device (str)**: the PyTorch device you want to use to inference the embedding model.
  * **embedding_model (str)**: name of the sentence_transformers embedding model you want to use to embed
* `ingest`: ingest data into the database. chops, chunks, and embeds input data for you. Saves the database to disk after ingestion. Returns nothing.
  * **text (str)**: the text you want to ingest into the database
  * **max_seq_length (int)**: max sequence length you want for your text chunks. default to 512
  * **metadata (dict)**: dictionary of metadata you want to associate with your entry
* `retrieve`: retrieve data from the database given input text. embeds input text and retrieves from database via cosine similarity. Returns a dictinoary with keys "texts" and optionally "metadata" and "scores" for each result.
  * **text (str)**: the text you want to use to retrieve from the database
  * **top_k (int)**: number of results you want to retrieve. default to 3
  * **autocut (bool)**: if we want to autocut our results or not. Will return the **top_k** *clusters* of results if this is true. More details on autocut above. Default to False.
  * **autocut_amount (int)**: if **autocut** is true, we retrieve **autocut_amount** of entries from the database instead of **top_k**. Default to 25.
  * **get_metadata (bool)**: if this is true, add metadata for each entry to response dictionary. Default to False.
  * **get_similarities (bool)**: if this is true, add similarity scores for each entry to response dictionary. Default to False.
  * **progress (bool)**: if this is true, we display retrieval progress bar. Default to False.
* `save`: save the database to disk. Saves vector index in the `.index` file, and all extra information (texts and metadata) in the `.info` file.
* `get_metadata_file`: returns the filename where metadata is located.
* `get_index_file`: returns the filename where the vector index is located.
* `get_metadata`: returns the metadata from the database as a dict in the format ID -> metadata.
  * **keys (list of integers)**: If you input keys, we will only return the metadata associated with the given keys. Default to empty list, which returns all metadata.
* `get_texts`: returns the texts from the database as a dict in the format ID -> text.
  * **keys (list of integers)**: If you input keys, we will only return the texts associated with the given keys. Default to empty list, which returns all texts.
* `get_vectors`: returns the vectors from the database as a dict in the format ID -> numpy vector.
  * **keys (list of integers)**: If you input keys, we will only return the texts associated with the given keys. Default to empty list, which returns all vectors.
* `clear`: clears the databse of all entries, making it as if you just instantiated. *This function WIPES your data*.
* `__len__`: get the length of the database. Calculated by the number of entries in the database.
* `__str__`: returns a stringified representation of the db, containing the name, metadata file, index file, and length.

## Class `Ingestor`
Utility class for ingesting data into VLite2. Found in [ingestors.py](vlite2/ingestors.py). Imported via `from vlite2 import Ingestor`.

* `__init__`: constructor for Ingestor class.
  * **vlite2 (VLite2 object)**: vlite2 object you want to ingest data *into*.
* `processString`: given a string, ingest it into the database after cleaning it up a bit.
  * **data (str)**: the string you want to ingest
  * **chunk_size (int)**: the chunk size you want when ingesting. Default to 128.
  * **source (str)**: source of the data you want to print if **verbose**. Default to "string".
  * **verbose (bool)**: if we want to print progress or not. Default to False.
* `processPDF`: ingest all the *text* from a PDF into the database.
  * **filename (str)**: .pdf filename you want to ingest into the database.
  * **chunk_size (int)**: the chunk size you want when ingesting. Default to 128.
  * **verbose (bool)**: if we want to print progress or not. Default to False.
* `processTXT`: ingest all the text from a TXT file into the database.
  * **filename (str)**: .txt filename you want to ingest into the database.
  * **chunk_size (int)**: the chunk size you want when ingesting. Default to 128.
  * **verbose (bool)**: if we want to print progress or not. Default to False.
* `processDict`: ingest data from a dictionary object into the database. Turns the dictionary into YAML format to better work with LLMs, then ingests it as a string. If you have multiple dictionaries, or a list of dictionaries, you should ingest them one by one with this method.
  * **data (dict)**: The data dictinoary you'd like to ingest to the database.
  * **chunk_size (int)**: the chunk size you want when ingesting. Default to 128.
  * **verbose (bool)**: if we want to print progress or not. Default to False.
* `processDocx`: ingest all the *text* from a word doc into the database. Also extracts images from the word doc, and saves them to a separate folder.
  * **filename (str)**: .docx filename you want to ingest into the database.
  * **chunk_size (int)**: the chunk size you want when ingesting. Default to 128.
  * **verbose (bool)**: if we want to print progress or not. Default to False.

## Class `EmbeddingModel`
EmbeddingModel class used to embed text into vectors. Found in [model.py](vlite2/model.py). Import via `from vlite2 import EmbeddingModel`. 

* `__init__`: constructor for EmbeddingModel class.
  * **model_name (str)**: the sentence_transformers model name you'd like to use when embedding. Uses huggingface's AutoTokenizer and AutoModel to instantiate tokenizers and models with this name. 
* `embed`: returns numpy vector embeddings for the data you'd like to embed!
  * **texts (list of strings)**: the list of texts you want to embed.
  * **max_seq_length (int)**: the max sequence length you want to use when tokenizing. Default to 256.
  * **device (str)**: the PyTorch device you want to use to inference the embedding model. Default to "mp3". 
* `token_count`: counts the number of tokens the tokenizer will generate in a body of input text.
  * **texts (list of strings)**: list of texts you want to compute tokens of.

# Notes
* Do not change the internal variables; many of those are used to track indexing between the `.index` and `.info` files, so unexpected behavior may arise
* The input types for both `ingest` and `retrieve` are text
  * Upon ingest, your text will be automatically chopped and chunked, embedded/vectorized, and saved to the database
  * Upon retrieve, your text will be automatically embedded/vectorized, and results will be retrieved from the vector index

# Pip Deploy
1. Delete existing `dist` and `build`.
2. `python3 setup.py sdist bdist_wheel`
3. `twine upload dist/*`

# Future Improvements
* Save `.info` and `.index` files in one, or better linking / state sharing between the two (hidden folder?)
* Better system to tracking and incrementing ID
