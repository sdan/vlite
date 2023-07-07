import time
import numpy as np
import pandas as pd
import json
import os
import chromadb
from chromadb.utils import embedding_functions
import cProfile
from pstats import Stats

import pinecone
from sentence_transformers import SentenceTransformer
import tqdm

import requests
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlite import main, utils
from vlite.main import VLite
from vlite.utils import load_file, chop_and_chunk, token_count


def main(query, corpuss, top_k, token_counts) -> pd.DataFrame:
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

    for corpus_idx, corpus in enumerate(corpuss):
        token_count = token_counts[corpus_idx]
        print(f"Running benchmarks for corpus of size {token_count}.")

        #################################################
        #                  VLite                        #
        #################################################
        if os.path.exists('vlite.npz'):
            print("[+] Removing vlite.npz")
            os.remove('vlite.npz')
        print("Begin VLite benchmark.")
        print("Adding vectors to VLite instance...")
        # with cProfile.Profile() as pr:

        t0 = time.time()

        vlite = VLite()
        try:
            _, vecs = vlite.memorize(corpus)
        except Exception as e:
            print(e)
            continue

        t1 = time.time()

        print(f"Took {t1 - t0:.3f}s to add vectors.")
        indexing_times.append(
            {
                "num_tokens": token_count,
                "lib": "VLite",
                "num_embeddings": vecs.shape[0],
                "indexing_time": t1 - t0,
            })

        print("Starting VLite trials...")

        times = []
        for i in range(len(query)):
            query_vector = query[i]
            t0 = time.time()
            try:
                texts, top_sims = vlite.remember(query_vector, top_k=top_k)
            except Exception as e:
                print(e)
                continue
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
        try:
            collection.add(documents=corpus, ids=ids)
        except Exception as e:
            print(e)
            print("Failed to add documents to Chroma collection.")
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

        for i in range(len(query)):
            query_vector = query[i]
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

        #################################################
        #                 Pinecone                      #
        #################################################
        print("Begin Pinecone benchmark.")
        print("Initializing Pinecone...")
        t0 = time.time()

                    environment="us-east-1-aws")
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
            try:
                xq = index.query(query_vector, top_k=5, include_metadata=True)
            except Exception as e:
                print("Exception: ", e)
                t0 = time.time()
                continue
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
        print("Begin tinyvector benchmark.")
        print("Initializing tinyvector instance...")
        base_url = "http://localhost:5001"
        table_name = "test"
        index_types = "brute_force"
        vector_dim = 384
        t0 = time.time()
        print("Creating tinyvector instance")
        response = requests.post(f"{base_url}/create_table", json={
            "table_name": table_name,
            "index_types": index_types,
            "dimension": 384,
            "use_uuid": True
        })
        print("Status code: ", response.status_code)
        print("Creating tinyvector index")
        try:
            response = requests.post(f"{base_url}/create_index", json={
                "table_name": table_name,
                "index_types": index_types
            })
            print("Status code: ", response.status_code)
        except:
            print("Error tinyvector index already exists")
            t0 = time.time()

        t1 = time.time()
        print(f"Took {t1 - t0:.3f}s to initialize")

        print("Adding vectors to tinyvector instance...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        t0 = time.time()
        for i in range(len(corpus)):
            embeddings = model.encode([corpus[i]]).tolist()
            print("[tinyvec] vector: ", i)
            print("[tinyvec] vector: ", embeddings[0])
            print("[tinyvec] corpus: ", corpus[i])
            try:
                response = requests.post(f"{base_url}/insert", json={
                    "table_name": table_name,
                    "ids": [i],
                    "vectors": embeddings[0],
                    "content": corpus[i]
                })
            except:
                print("Error inserting vector")
                t0 = time.time()
                break

        t1 = time.time()
        print(f"Took {t1 - t0:.3f}s to add vectors.")
        indexing_times.append(
            {
                "num_tokens": token_count,
                "lib": "tinyvector",
                "num_embeddings": len(corpus),
                "indexing_time": t1 - t0,
            }
        )

        # no query for now

        #################################################
        #                  qdrant                       #
        #################################################
        qdrant_client = QdrantClient(
            ":memory:", # fast, wraps numpy

            # persistent, scalable
            # url="<your-qdrant-instance-url-here>",
            # api_key="<your-api-key-here>",
        )

        qdrant_client.recreate_collection(
            collection_name="my_collection",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print("Begin Qdrant benchmark.")
        print("Adding vectors to Qdrant instance...")

        t0 = time.time()

        model = SentenceTransformer('all-MiniLM-L6-v2')
        vectors = model.encode(corpus).tolist()

        try:
            qdrant_client.upsert(
                collection_name="my_collection",
                points=[
                    PointStruct(
                        id=idx,
                        vector=model.encode(vector).tolist(),
                        payload={"text": corpus[idx]}
                    )
                    for idx, vector in enumerate(corpus)
                ]
            )
        except Exception as e:
            print(e)
            print("Failed to upsert vectors to Qdrant instance.")
            t0 = time.time()

        t1 = time.time()

        print(f"Took {t1 - t0:.3f}s to add vectors.")
        indexing_times.append(
            {
                "num_tokens": token_count,
                "lib": "Qdrant",
                "num_embeddings": len(vectors),
                "indexing_time": t1 - t0,
            })

        print("Starting Qdrant trials...")

        times = []
        for i in range(len(query)):
            query_vector = model.encode(query[i]).tolist()
            t0 = time.time()
            try:
                hits = qdrant_client.search(
                    collection_name="my_collection",
                    query_vector=query_vector,
                    limit=top_k  # Return top_k closest points
                )
                # print("Top hits: ", hits)
                t1 = time.time()
            except Exception as e:
                print(e)
                print("Failed to query Qdrant instance.")
                t0 = time.time()
            times.append(t1 - t0)

        print(f"Took {t1 - t0:.3f}s to query.")

        results.append(
            {
                "num_embeddings": len(vectors),
                "lib": "Qdrant",
                "k": top_k,
                "avg_time": np.mean(times),
                "stddev_time": np.std(times),
            }
        )

        print(json.dumps(results[-1], indent=2))
        print("Done Qdrant")

        #################################################
        #                  milvus                       #
        #################################################

        # too complicated docs
        temp_results = pd.DataFrame(results)
        temp_indexing_times = pd.DataFrame(indexing_times)
        temp_results.to_csv("temp_vlite_benchmark_query.csv", index=False)
        temp_indexing_times.to_csv("temp_vlite_benchmark_index.csv", index=False)

    results = pd.DataFrame(results)
    indexing_times = pd.DataFrame(indexing_times)
    return results, indexing_times


if __name__ == "__main__":
    # Benchmark Config
    k = 10
    queries = [
        "Transformer-based language model architecture",
        "Training large-scale language models",
        "Performance evaluation of GPT-4",
        "Applications of GPT-4 in natural language processing",
        "Improvements in GPT-4 over previous versions",
        "Understanding the architecture of GPT-4",
        "GPT-4's impact on artificial intelligence research",
        "The role of attention mechanisms in GPT-4",
        "How GPT-4 handles long-range dependencies in text",
        "The scalability of GPT-4 model training",
        "The use of GPT-4 in generating human-like text",
        "The influence of GPT-4 on future AI models",
        "The limitations and challenges of training GPT-4",
        "The role of GPT-4 in advancing conversational AI",
        "Understanding the tokenization process in GPT-4",
        "The computational resources required for training GPT-4",
        "The ethical considerations of deploying GPT-4",
        "The impact of GPT-4 on machine translation tasks",
        "How GPT-4 handles context in language understanding",
        "The role of pre-training and fine-tuning in GPT-4",
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

    benchmark_corpuss = [chopped_corpus, chopped_corpus*2, chopped_corpus*4, chopped_corpus*16, chopped_corpus*64]
    benchmark_token_counts = [token_count, token_count*2, token_count*4, token_count*16, token_count*64]

    print("token count", token_count)
    print("corp len", len(chopped_corpus))

    results, indexing_times = main(queries, benchmark_corpuss, k, benchmark_token_counts)
    print(results)
    print(indexing_times)
    results.to_csv("vlite_benchmark_query.csv", index=False)
    indexing_times.to_csv("vlite_benchmark_index.csv", index=False)
