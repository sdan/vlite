import time
import numpy as np
import pandas as pd
import json
import os

from vlite import main, utils
from vlite.main import VLite
from vlite.utils import load_file, chop_and_chunk
from vlite.model import EmbeddingModel

import chromadb
from chromadb.utils import embedding_functions

def main(queries, corpuss, top_k, token_counts) -> pd.DataFrame:
    """Run the VLite benchmark.

    Parameters
    ----------
    queries : list
        A list of query strings to test the retrieval performance.
    corpuss : list
        A list of text corpuses of different sizes to benchmark indexing and retrieval.
    top_k : int
        The number of top similar results to retrieve for each query.
    token_counts : list
        A list of token counts corresponding to each corpus size.

    Returns
    -------
    results : pd.DataFrame
        A DataFrame containing results from the VLite benchmark.
    indexing_times : pd.DataFrame
        A DataFrame containing indexing times for each corpus size.
    """
    results = []
    indexing_times = []

    for corpus_idx, corpus in enumerate(corpuss):
        token_count = token_counts[corpus_idx]
        print(f"Running benchmarks for corpus of size {token_count} tokens.")

        #################################################
        #                  VLite                        #
        #################################################
        
        if os.path.exists('vlite-bench.npz'):
            print("[+] Removing vlite-bench.npz")
            os.remove('vlite-bench.npz')

        print("Adding vectors to VLite instance...")
        t0 = time.time()

        vlite = VLite("vlite-bench")
        result_add = vlite.add(corpus)
        t1 = time.time()

        print(f"Took {t1 - t0:.3f}s to add {len(result_add)} vectors.")
        indexing_times.append(
            {
                "num_tokens": token_count,
                "lib": "VLite",
                "num_embeddings": len(result_add),
                "indexing_time": t1 - t0,
            })

        print("Starting VLite retrieval trials...")
        results_retrieve = []
        times = []
        for query in queries:
            t0 = time.time()
            results_retrieve = vlite.retrieve(query, top_k=top_k)
            t1 = time.time()
            times.append(t1 - t0)

            print(f"Top {top_k} results for query '{query}':")
            for text, similarity, metadata in results_retrieve:
                print(f"Text: {text}\nSimilarity: {similarity}\nMetadata: {metadata}\n---")

        results.append(
            {
                "num_embeddings": len(results_retrieve),
                "lib": "VLite",
                "k": top_k,
                "avg_time": np.mean(times),
                "stddev_time": np.std(times),
            }
        )

        print(json.dumps(results[-1], indent=2))
        print("Done VLite benchmark.")
        
        #################################################
        #                  Chroma                       #
        #################################################
        print("Begin Chroma benchmark.")
        print("Adding vectors to Chroma instance...")
        t0 = time.time()

        chroma_client = chromadb.Client()
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="mixedbread-ai/mxbai-embed-large-v1")
        collection = chroma_client.create_collection(name="my_collection", embedding_function=sentence_transformer_ef)        
        ids = [str(i) for i in range(len(corpus))]
        try:
            collection.add(documents=corpus, ids=ids)
        except Exception as e:
            print(e)
            print("Failed to add documents to Chroma collection.a")
            t0 = time.time()

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

        for query_vector in queries:
            t0 = time.time()
            try:
                chroma_results = collection.query(
                    query_texts=[query_vector], n_results=top_k)
            except Exception as e:
                print(e)
                print("Failed to query Chroma collection.")
                t0 = time.time()
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

    results_df = pd.DataFrame(results)
    indexing_times_df = pd.DataFrame(indexing_times)
    return results_df, indexing_times_df


if __name__ == "__main__":
    # Benchmark Config
    k = 5
    queries = [
        "What is the architecture of transformers?",
        "How does transformers handle contextual understanding?",
        "What are the key improvements in LSTM over Transformers?",
        "How many parameters does Transformers have?",
        "What are the limitations of Transformers?",
    ]
    corpus = load_file(os.path.join(os.path.dirname(__file__), 'data/attention.pdf'))
    chopped_corpus = chop_and_chunk(text=corpus)
    token_count = EmbeddingModel().token_count(" ".join(chopped_corpus))

    # benchmark_corpuss = [chopped_corpus, chopped_corpus*2, chopped_corpus*4]
    # benchmark_token_counts = [token_count, token_count*2, token_count*4]
    
    benchmark_corpuss = [chopped_corpus]
    benchmark_token_counts = [token_count]

    print("Token count:", token_count)
    print("Corpus length:", len(chopped_corpus))

    results, indexing_times = main(queries, benchmark_corpuss, k, benchmark_token_counts)
    print("Benchmark Results:")
    print(results)
    print("Indexing Times:")
    print(indexing_times)
    results.to_csv("vlite2_benchmark_results.csv", index=False)
    indexing_times.to_csv("vlite2_benchmark_indexing_times.csv", index=False)