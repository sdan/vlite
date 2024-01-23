# VLite V2

V2 is a collection and implementation of improvements upon the original amazing [VLite](https://github.com/sdan/vlite). Blazing fast and with even more features.
All written in < 200 lines of Python! Download via `pip install vlite2`. PyPi project found [here](https://pypi.org/project/vlite2/). You can find benchmarking
code at this repo [here](https://github.com/raydelvecchio/vdb-benchmark).

# Implemented V2 Improvements

I had been using this for a few projects and found I had some more use cases not in the original. New features are
as follows:

* VLite now allows you to remember metadata associated with topk texts, allowing it to be used as a semantic 
search key-value database
* All important internal variables are private for enhanced safety and security; only device is left exposed since that can be changed on use case
* Significantly less dependencies, and thus reduced install time, since benchmarking and testing found [elsewhere](https://github.com/raydelvecchio/vdb-benchmark).
* Can now set max sequence length for chunking for each independent memorize operation; great for more varied data
across a given database
* When chunking, removed default newline split return; the db will now chunk ANY text, even if it already contains
some newlines (\n) within it
* **Naive implementation of Weaviate's [autocut](https://weaviate.io/developers/weaviate/search/hybrid#limiting-results-with-autocut-and-auto_limit)!** In the 
`remember` function, we no longer need to set K, rather, clusters of results are returned with each other.
  * I implement this by first finding the differences in similarity between the top X (hyperparameter) scores, calculating
  the standard deviation, then clustering by differences LARGER than the standard deviation. Results are returned
  then based on the number of CLUSTERS you want to reference, rather than direct elements.
  * If K number of relevant clusters do NOT exist, then the maximum number of clusters possible is returned by default
  * EX: `[10, 9, 4, 3, -2]` -> `[10, 9], [4, 3], [-2]`, if `k=2` we return elements associated with differences `[10, 9, 4, 3]`
* Integration with [USearch](https://github.com/unum-cloud/usearch) as the vector search engine instead of custom numpy comparisons
  * USearch handles search, vector computation, and indexing natively as the fastest vector search engine in the world!
  * Documentation for using USearch found [here](https://unum-cloud.github.io/usearch/python/reference.html#usearch.index.Index.add)!

# Pip Deploy
1. Delete existing `dist` and `build`.
2. `python3 setup.py sdist bdist_wheel`
3. `twine upload dist/*`

# Future Improvements
* Save `.info` and `.index` files in one, or better linking / state sharing between the two (hidden folder?)
* Better system to tracking and incrementing ID
