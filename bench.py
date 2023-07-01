import time
import numpy as np
import pandas as pd
import json
from main import VLite
import hnswlib
def main(num_dims: int, k: int, num_trials: int, size_range: list[int]) -> pd.DataFrame:
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
    for num_embeddings in size_range:
        print(f"Creating sample data of {num_embeddings:,} {num_dims}-dimensional vectors")
        data = np.random.rand(num_embeddings, num_dims)

        #################################################
        #                  VLite                        #
        #################################################
        print("Begin VLite benchmark.")
        print("Adding vectors to VLite instance...")
        t0 = time.time()

        vlite = VLite()
        for vector in data:
            vlite.add_vector(vector)

        t1 = time.time()
        print(f"Took {t1 - t0:.3f}s to add vectors.")

        print("Starting VLite trials...")
        times = []
        for i in range(num_trials):
            query_vector = data[i]
            t0 = time.time()
            vlite.get_similar_vectors(query_vector, top_k=k)
            t1 = time.time()
            times.append(t1 - t0)

        results.append(
            {
                "num_embeddings": num_embeddings,
                "num_dims": num_dims,
                "lib": "VLite",
                "k": k,
                "avg_time": np.mean(times),
                "stddev_time": np.std(times),
            }
        )
        print(json.dumps(results[-1], indent=2))
        print("Done VLite")

        #################################################
        #                  Chroma                       #
        #################################################
        

        #################################################
        #                 Pinecone                      #
        #################################################

        #################################################
        #                  pgvector                     #
        #################################################

        #################################################
        #                  HNSW                         #
        #################################################
        print("Begin hnswlib benchmark.")
        print("Building index...")
        t0 = time.time()

        # NOTE: Much of the below hnswlib code was just taken from the hnswlib README.

        # NOTE: We choose inner product since this is equivalent to cosine if you
        # normalize vectors prior to insertion.
        p = hnswlib.Index(
            space="ip", dim=num_dims
        )  # possible options are l2, cosine or ip

        # NOTE: Use default settings from the README.
        p.init_index(max_elements=num_embeddings, ef_construction=200, M=16)
        ids = np.arange(num_embeddings)
        p.add_items(data, ids)
        p.set_ef(50)

        t1 = time.time()
        print(f"Took {t1 - t0:.3f}s to build index.")

        print("Starting hnswlib trials...")
        times = []
        for i in range(num_trials):
            t0 = time.time()
            p.knn_query(data[[i], :], k=k)
            t1 = time.time()
            times.append(t1 - t0)

        results.append(
            {
                "num_embeddings": num_embeddings,
                "num_dims": num_dims,
                "lib": "hnswlib",
                "k": k,
                "avg_time": np.mean(times),
                "stddev_time": np.std(times),
            }
        )
        print(json.dumps(results[-1], indent=2))

        print("Done hnswlib")


    results = pd.DataFrame(results)
    return results

if __name__ == "__main__":
    # Benchmark Config
    num_dims = 384
    num_trials = 5
    size_range = [1_000, 2_000, 5_000]
    k = 10

    results = main(num_dims, k, num_trials, size_range)
    print(results)
    results.to_csv("vlite_benchmark_results.csv", index=False)
