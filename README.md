# VLite V2

Improvements upon [VLite](https://github.com/sdan/vlite). Still blazing fast, and still in Numpy, but with
many more features. You can find the original information about the project from Surya [here](https://twitter.com/sdand/status/1676256437918633984).
It's still faster than other VDBs.

# About Section (Original)

VLite is a vector database built for agents, ChatGPT Plugins, and other AI apps that need a fast and simple database to store vectors. 

I (Surya) built it to support the millions of embeddings I generate , index, and sort with [ChatWith+ ChatGPT Plugins](https://plugins.sdan.io/) which run for millions of users. Most vector databases either repeatedly crashed on a daily basis or was too expensive for the throughput I was putting through.

It uses Apple's Metal Performance Shaders via Pytorch to accelerate vector loading and uses CPU threading to accelerate vector queries to reduce time spent copying vectors from the GPU(MPS) to the CPU.

# V2 Improvements

I (Ray) had been using this for a few projects and found I had some more use cases not in the original. New features are
as follows:

* VLite now allows you to remember metadata associated with topk texts, allowing it to be used as a semantic 
search key-value database
* Can now set max sequence length for chunking for each independent memorize operation; great for more varied data
across a given database
* When chunking, removed default newline split return; the db will now chunk ANY text, even if it already contains
some newlines (\n) within it
* Naive implementation of Weaviate's [autocut](https://weaviate.io/developers/weaviate/search/hybrid#limiting-results-with-autocut-and-auto_limit)! In the 
`remember` function, we no longer need to set K, rather, clusters of results are returned with each other.