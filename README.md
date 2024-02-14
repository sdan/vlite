# VLite V2 (VLite2)

V2 is a collection and implementation of improvements upon the original amazing [VLite](https://github.com/sdan/vlite). Blazing fast and with even more features.
All written in a few hundred lines of Python! Download via `pip install vlite2`. PyPi project found [here](https://pypi.org/project/vlite2/). You can find benchmarking
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

# Implemented V2 Improvements

I had been using this for a few projects and found I had some more use cases not in the original. New features are
as follows:

* VLite now allows you to retrieve metadata associated with topk texts, allowing it to be used as a semantic 
search key-value database
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

# Documentation
Does not exist yet. Please refer to [main.py](vlite2/main.py) to see how to use each method and parameters.

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
