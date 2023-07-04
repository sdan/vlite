import time
import numpy as np
import pandas as pd
import json
from main import VLite
from utils import load_file, chop_and_chunk, token_count
import os
import chromadb
from chromadb.utils import embedding_functions
import cProfile
from pstats import Stats

import pinecone
from sentence_transformers import SentenceTransformer
import tqdm

def main(query, corpus, top_k, token_count) -> pd.DataFrame:
    """Run the benchmarks.

    Parameters
    ----------
    num_dims : The number of dimensions each embedding should have.
    k : The number of closest embeddings to search for.
    num_trials : The number of trials to run for each benchmark.
    size_range : A list containing the numbers of embeddings to run benchmarks for.

    Returns
    -------
    results : A DataFrame containing results from all benchmarks.
    """
    results = []
    indexing_times = []


    #################################################
    #                  VLite                        #
    #################################################
    if os.path.exists('vlite.pkl'):
        print("[+] Removing vlite.pkl")
        os.remove('vlite.pkl')
    print("Begin VLite benchmark.")
    print("Adding vectors to VLite instance...")
    # with cProfile.Profile() as pr:
    
    t0 = time.time()

    vlite = VLite()
    _, vecs = vlite.memorize(corpus)

    t1 = time.time()

    # stats = Stats(pr)
    # stats.strip_dirs().sort_stats("time").print_stats()

    print(f"Took {t1 - t0:.3f}s to add vectors.")
    indexing_times.append(
        {
            "num_tokens": token_count,
            "lib": "VLite",
            "num_embeddings": vecs.shape[0],
            "indexing_time": t1 - t0,
        })

    print("Starting VLite trials...")
    
    # with cProfile.Profile() as pr:

    times = []
    for i in range(len(query)):
        query_vector = query[i]
        t0 = time.time()
        texts, top_sims = vlite.remember(query_vector, top_k=top_k)
        print(top_sims)
        # print(f"Top {top_k} sims: {top_sims}")
        # print(f"Top {top_k} texts: {texts}")
        t1 = time.time()
        times.append(t1 - 
                     t0)

    results.append(
        {
            "num_embeddings": vecs.shape[0],
            "lib": "VLite",
            "k": top_k,
            "avg_time": np.mean(times),
            "stddev_time": np.std(times),
        }
    )

    print(json.dumps(results[-1], indent=2))
    print("Done VLite")

    #################################################
    #                  Chroma                       #
    #################################################
    print("Begin Chroma benchmark.")
    print("Adding vectors to Chroma instance...")
    t0 = time.time()

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="my_collection")
    # generate list of ids that is the length of the corpus
    ids = [str(i) for i in range(len(corpus))]
    collection.add(documents=corpus, ids=ids)

    t1 = time.time()
    print(f"Took {t1 - t0:.3f}s to add vectors.")
    indexing_times.append(
        {
            "num_tokens": token_count,
            "lib": "Chroma",
            "num_embeddings": len(corpus),
            "indexing_time": t1 - t0,
        }
    )

    print("Starting Chroma trials...")
    times = []

    for i in range(len(query)):
        query_vector = query[i]
        t0 = time.time()
        chroma_results = collection.query(query_texts=[query_vector], n_results=top_k)
        print(f"Top {top_k} results: {chroma_results['distances']}")
        t1 = time.time()
        times.append(t1 - t0)
    
    results.append(
        {
            "num_embeddings": len(corpus),
            "lib": "Chroma",
            "k": top_k,
            "avg_time": np.mean(times),
            "stddev_time": np.std(times),
        }
    )
    #################################################
    #                 Pinecone                      #
    #################################################
    print("Begin Pinecone benchmark.")
    print("Initializing Pinecone...")
    t0 = time.time()

    pinecone.init(api_key="1de0e65b-6645-4139-a8c3-4b8b5f1dfdb0", environment="us-east-1-aws")
    index_name = "quickstart"

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=384, metric="cosine")

    # now connect to the index
    index = pinecone.Index(index_name)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    t1 = time.time()
    print(f"Took {t1 - t0:.3f}s to initialize")

    t0 = time.time()
    print("Adding vectors to Pinecone instance...")
    batch_size = 128
    for i in tqdm.tqdm(range(0, len(corpus), batch_size)):
        print("Finding end of batch")
        # find end of batch
        i_end = min(i+batch_size, len(corpus))
        print("Creating IDs batch")
        # create IDs batch
        ids = [str(x) for x in range(i, i_end)]
        # create metadata batch
        metadatas = [{'text': text} for text in corpus[i:i_end]]
        print("Creating embeddings")
        # create embeddings
        embeddings = model.encode(corpus[i:i_end]).tolist()
        print("Creating records list for upsert")
        # create records list for upsert
        # print("Zipping IDs and embeddings")
        # print("IDs: ", ids)
        # print("Embeddings: ", embeddings)
        # print("Metadatas: ", metadatas)
        records = zip(ids, embeddings, metadatas)
        print("Upserting to Pinecone")
        # upsert to Pinecone
        index.upsert(vectors=records)

    t1 = time.time()
    print(f"Took {t1 - t0:.3f}s to add vectors.")
    indexing_times.append(
        {
            "num_tokens": token_count,
            "lib": "Pinecone",
            "num_embeddings": len(corpus),
            "indexing_time": t1 - t0,
        }
    )


    # Query Pinecone
    print("Starting Pinecone trials...")
    times = []
    for query_text in query:
        # create the query vector
        query_vector = model.encode([query_text]).tolist()
        # now query
        t0 = time.time()
        xq = index.query(query_vector, top_k=5, include_metadata=True)
        t1 = time.time()
        times.append(t1 - t0)
        
        print(f"Top 5 results for '{query_text}'")
        for result in xq['matches']:
            print(f"{round(result['score'], 2)}")

    print("results: ", len(corpus))
    print("times: ", times)
    print("top_k: ", top_k)

    results.append(
        {
            "num_embeddings": len(corpus),
            "lib": "Pinecone",
            "k": top_k,
            "avg_time": np.mean(times),
            "stddev_time": np.std(times),
        }
    )

    #################################################
    #                  tinyvector                   #
    #################################################

    #################################################
    #                  qdrant                       #
    #################################################

    #################################################
    #                  milvus                       #
    #################################################


    results = pd.DataFrame(results)
    indexing_times = pd.DataFrame(indexing_times)
    return results, indexing_times

if __name__ == "__main__":
    # Benchmark Config
    k = 10
    queries = [
            "What is the architecture of GPT-4?",
            "How does GPT-4 handle contextual understanding?",
            "What are the key improvements in GPT-4 over GPT-3?",
            "How many parameters does GPT-4 have?",
            "What are the limitations of GPT-4?",
            "What datasets were used to train GPT-4?",
            "How does GPT-4 handle longer context?",
            "What is the computational requirement for training GPT-4?",
            "What techniques were used to train GPT-4?",
            "What is the impact of GPT-4 on natural language processing?",
            "What are the use cases demonstrated in the GPT-4 paper?",
            "What are the evaluation metrics used in GPT-4's paper?",
            "What kind of ethical considerations are discussed in the GPT-4 paper?",
            "How does the GPT-4 handle tokenization?",
            "What are the novel contributions of the GPT-4 model?"
        ]
    corpus = load_file('test-data/gpt-4.pdf')
    chopped_corpus = chop_and_chunk(text=corpus)
    token_count = token_count(chopped_corpus)
    print("token count", token_count)
    print("corp len", len(chopped_corpus))
    

    results, indexing_times = main(queries, chopped_corpus, k, token_count)
    print(results)
    print(indexing_times)
    results.to_csv("vlite_benchmark_results.csv", index=False)
    indexing_times.to_csv("vlite_benchmark_indexing_times.csv", index=False)
