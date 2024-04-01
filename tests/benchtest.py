import time
import numpy as np
import pandas as pd
import json
import os

from vlite import main, utils
from vlite.main import VLite
from vlite.utils import load_file, chop_and_chunk
from vlite.model import EmbeddingModel

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
        print(f"Running VLite benchmark for corpus of size {token_count} tokens.")

        if os.path.exists('vlite.npz'):
            print("[+] Removing vlite.npz")
            os.remove('vlite.npz')

        print("Adding vectors to VLite instance...")
        t0 = time.time()

        vlite = VLite()
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
    corpus = load_file('data/attention.pdf')
    chopped_corpus = chop_and_chunk(text=corpus)
    chopped_corpus = " ".join(chopped_corpus)
    token_count = EmbeddingModel().token_count(chopped_corpus)

    benchmark_corpuss = [chopped_corpus, chopped_corpus*2, chopped_corpus*4]
    benchmark_token_counts = [token_count, token_count*2, token_count*4]

    print("Token count:", token_count)
    print("Corpus length:", len(chopped_corpus))

    results, indexing_times = main(queries, benchmark_corpuss, k, benchmark_token_counts)
    print("Benchmark Results:")
    print(results)
    print("Indexing Times:")
    print(indexing_times)
    results.to_csv("vlite2_benchmark_results.csv", index=False)
    indexing_times.to_csv("vlite2_benchmark_indexing_times.csv", index=False)