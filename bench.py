import time
import numpy as np
import pandas as pd
import json
from main import VLite
from utils import load_file, chop_and_chunk
import os
import chromadb
from chromadb.utils import embedding_functions
import cProfile
from pstats import Stats


def main(query, corpus, top_k) -> pd.DataFrame:
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
        times.append(t1 - t0)

    results.append(
        {
            "num_embeddings": vecs.shape[0],
            "lib": "VLite",
            "k": top_k,
            "avg_time": np.mean(times),
            "stddev_time": np.std(times),
        }
    )

        # stats = Stats(pr)
        # stats.strip_dirs().sort_stats("time").print_stats()


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

    #################################################
    #                  pgvector                     #
    #################################################

    results = pd.DataFrame(results)
    return results

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
    corpus = chop_and_chunk(text=corpus)

    print("corp chop", corpus)
    print("corp len", len(corpus))
    

    results = main(queries, corpus, k)
    print(results)
    results.to_csv("vlite_benchmark_results.csv", index=False)
