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

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.document_loaders import TextLoader
from langchain.vectorstores import LanceDB, Lantern, FAISS
from langchain.text_splitter import CharacterTextSplitter

from langchain.docstore.document import Document

from langchain_community.vectorstores.pgvector import PGVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


from pinecone import Pinecone, ServerlessSpec

import tqdm

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
        if os.path.exists('omnoms/vlite-bench.omom'):
            print("[+] Removing vlite-bench.omom")
            os.remove('omnoms/vlite-bench.omom')

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
                "num_embeddings": len(result_add),
                "lib": "VLite",
                "k": top_k,
                "avg_time": np.mean(times),
                "stddev_time": np.std(times),
            }
        )

        print(json.dumps(results[-1], indent=2))
        print("Done VLite benchmark.")
        
                
        #################################################
        #                  LanceDB                      #
        #################################################
        print("Begin LanceDB benchmark.")
        print("Adding documents to LanceDB instance...")
        t0 = time.time()

        embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
        documents = [Document(page_content=text) for text in corpus]
        docsearch = LanceDB.from_documents(documents, embeddings)

        t1 = time.time()
        print(f"Took {t1 - t0:.3f}s to add documents.")
        indexing_times.append(
            {
                "num_tokens": token_count,
                "lib": "LanceDB",
                "num_embeddings": len(documents),
                "indexing_time": t1 - t0,
            }
        )

        print("Starting LanceDB trials...")
        times = []
        for query in queries:
            t0 = time.time()
            docs = docsearch.similarity_search(query, k=top_k)
            t1 = time.time()
            times.append(t1 - t0)

            print(f"Top {top_k} results for query '{query}':")
            for doc in docs:
                print(f"Text: {doc.page_content}\n---")

        results.append(
            {
                "num_embeddings": len(documents),
                "lib": "LanceDB",
                "k": top_k,
                "avg_time": np.mean(times),
                "stddev_time": np.std(times),
            }
        )

        print(json.dumps(results[-1], indent=2))
        print("Done LanceDB benchmark.")
        

        #################################################
        #                  pgvector       unable to run remote                #
        #################################################
        # print("Begin PGVector benchmark.")
        # print("Adding documents to PGVector instance...")
        # t0 = time.time()

        # embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

        # documents = [Document(page_content=text.replace('\0', '')) for text in corpus]

        # CONNECTION_STRING = "postgresql+psycopg2://postgres:password@localhost:5433/embedding_test"

        # db = PGVector.from_documents(
        #     embedding=embeddings,
        #     documents=documents,
        #     collection_name="benchmark_test",
        #     connection_string=CONNECTION_STRING,
        # )

        # t1 = time.time()
        # print(f"Took {t1 - t0:.3f}s to add documents.")
        # indexing_times.append(
        #     {
        #         "num_tokens": token_count,
        #         "lib": "PGVector",
        #         "num_embeddings": len(documents),
        #         "indexing_time": t1 - t0,
        #     }
        # )

        # print("Starting PGVector trials...")
        # times = []
        # for query in queries:
        #     t0 = time.time()
        #     docs = db.similarity_search_with_score(query, k=top_k)
        #     t1 = time.time()
        #     times.append(t1 - t0)

        #     print(f"Top {top_k} results for query '{query}':")
        #     for doc, score in docs:
        #         print(f"Text: {doc.page_content}\nScore: {score}\n---")

        # results.append(
        #     {
        #         "num_embeddings": len(documents),
        #         "lib": "PGVector",
        #         "k": top_k,
        #         "avg_time": np.mean(times),
        #         "stddev_time": np.std(times),
        #     }
        # )

        # print(json.dumps(results[-1], indent=2))
        # print("Done PGVector benchmark.")

        #################################################
        #                  FAISS                        #
        #################################################
        print("Begin FAISS benchmark.")
        print("Adding documents to FAISS instance...")
        t0 = time.time()

        embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
        documents = [Document(page_content=text) for text in corpus]
        db = FAISS.from_documents(documents, embeddings)

        t1 = time.time()
        print(f"Took {t1 - t0:.3f}s to add documents.")
        indexing_times.append(
            {
                "num_tokens": token_count,
                "lib": "FAISS",
                "num_embeddings": len(corpus),
                "indexing_time": t1 - t0,
            }
        )

        print("Starting FAISS trials...")
        times = []
        for query in queries:
            t0 = time.time()
            docs = db.similarity_search(query, k=top_k)
            t1 = time.time()
            times.append(t1 - t0)

            print(f"Top {top_k} results for query '{query}':")
            for doc in docs:
                print(f"Text: {doc.page_content}\n---")

        results.append(
            {
                "num_embeddings": len(corpus),
                "lib": "FAISS",
                "k": top_k,
                "avg_time": np.mean(times),
                "stddev_time": np.std(times),
            }
        )

        print(json.dumps(results[-1], indent=2))
        print("Done FAISS benchmark.")

        #################################################
        #                  Chroma                       #
        #################################################
        print("Begin Chroma benchmark.")
        print("Adding vectors to Chroma instance...")
        t0 = time.time()

        chroma_client = chromadb.Client()
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="mixedbread-ai/mxbai-embed-large-v1")
        collection = chroma_client.get_or_create_collection(name="benchtest", embedding_function=sentence_transformer_ef)        
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

    #     #################################################
    #     #                  Pinecone   unable to run remote                  #
    #     #################################################
    #     print("Begin Pinecone benchmark.")
    #     print("Initializing Pinecone...")
    #     t0 = time.time()

    #     # Replace these with your Pinecone API key and environment
    #     pinecone = Pinecone(api_key="")


    #     environment = "us-east-1-aws"
    #     index_name = "quickstart"

    # #     if index_name not in pinecone.list_indexes():
    # #         pinecone.create_index(name=index_name, dimension=1024, metric="euclidean", # Replace with your model metric
    # # spec=ServerlessSpec(
    # #     cloud="aws",
    # #     region="us-west-2"
    # # ))
    #     t1 = time.time()
    #     # now connect to the index
    #     index = pinecone.Index("quickstart")
    #     model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')

        
        

    #     t0 = time.time()
    #     print(f"Took {t1 - t0:.3f}s to initialize")
    #     print("Adding vectors to Pinecone instance...")
    #     batch_size = 128

    #     for i in tqdm.tqdm(range(0, len(corpus), batch_size)):
    #         # find end of batch
    #         i_end = min(i + batch_size, len(corpus))
    #         # create IDs batch
    #         ids = [str(x) for x in range(i, i_end)]
    #         # create metadata batch
    #         metadatas = [{'text': text} for text in corpus[i:i_end]]
    #         # create embeddings
    #         embeddings = model.encode(corpus[i:i_end]).tolist()
    #         # create records list for upsert
    #         records = zip(ids, embeddings, metadatas)
    #         # upsert to Pinecone
    #         index.upsert(vectors=records)

    #     t1 = time.time()
    #     print(f"Took {t1 - t0:.3f}s to add vectors.")
    #     indexing_times.append(
    #         {
    #             "num_tokens": token_count,
    #             "lib": "Pinecone",
    #             "num_embeddings": len(corpus),
    #             "indexing_time": t1 - t0,
    #         }
    #     )

    #     # Query Pinecone
    #     print("Starting Pinecone trials...")
    #     times = []
    #     for query_text in queries:
    #         # create the query vector
    #         query_vector = model.encode([query_text]).tolist()[0]
    #         # now query
    #         t0 = time.time()
    #         try:
    #             xq = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    #         except Exception as e:
    #             print("Exception: ", e)
    #             t0 = time.time()
    #             continue
    #         t1 = time.time()
    #         times.append(t1 - t0)

    #         print(f"Top {top_k} results for '{query_text}'")
    #         for result in xq['matches']:
    #             print(f"{round(result['score'], 2)}: {result['metadata']['text'][:100]}...")

    #     results.append(
    #         {
    #             "num_embeddings": len(corpus),
    #             "lib": "Pinecone",
    #             "k": top_k,
    #             "avg_time": np.mean(times),
    #             "stddev_time": np.std(times),
    #         }
    #     )

    #     print(json.dumps(results[-1], indent=2))
    #     print("Done Pinecone benchmark.")


        
        #################################################
        #                  Qdrant                       #
        #################################################
        print("Begin Qdrant benchmark.")
        print("Adding vectors to Qdrant instance...")
        t0 = time.time()

        qdrant_client = QdrantClient(
            ":memory:"  # Use in-memory storage for benchmarking
        )
        qdrant_client.recreate_collection(
            collection_name="my_collection",
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

        model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
        vectors = model.encode(corpus).tolist()
        
        print("Adding vectors to Qdrant instance...")

        try:
            qdrant_client.upsert(
                collection_name="my_collection",
                points=[
                    PointStruct(
                        id=idx,
                        vector=model.encode(text).tolist(),
                        payload={"text": text}
                    )
                    for idx, text in enumerate(corpus)
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
            }
        )

        print("Starting Qdrant trials...")
        times = []
        
        for i in range(len(query)):
            t0 = time.time()
            query_vector = model.encode(query).tolist()
            try:
                hits = qdrant_client.search(
                    collection_name="my_collection",
                    query_vector=query_vector,
                    limit=top_k
                )
                # print(f"Top {top_k} results: {[hit.payload['text'] for hit in hits]}")
                t1 = time.time()
            except Exception as e:
                print(e)
                print("Failed to query Qdrant instance.")
                t0 = time.time()
            times.append(t1 - t0)

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
        print("Done Qdrant benchmark.")

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
        "What are the applications of Transformers?",
        "How do Transformers handle long-range dependencies?",
        "What is the difference between Transformers and RNNs?",
        "How does the attention mechanism work in Transformers?",
        "What are the different types of attention used in Transformers?",
        "How do Transformers handle parallelization?",
        "What are the common use cases of Transformers?",
        "How do Transformers perform on different NLP tasks?",
        "What are the challenges in training large Transformer models?",
        "How do Transformers handle multi-task learning?",
        "What are the recent advances in Transformer architecture?",
        "How do Transformers handle out-of-distribution data?",
        "What are the techniques for efficient inference with Transformers?",
        "How do Transformers handle multimodal data?",
        "What are the pros and cons of using Transformers compared to other models?"
    ]

    corpus = load_file(os.path.join(os.path.dirname(__file__), 'data/attention.pdf'))
    chopped_corpus = chop_and_chunk(text=corpus)
    token_count = EmbeddingModel().token_count(" ".join(chopped_corpus))

    benchmark_corpuss = [
        chopped_corpus,
        chopped_corpus * 2,
        chopped_corpus * 4,
        chopped_corpus * 8,
        chopped_corpus * 16
    ]

    benchmark_token_counts = [
        token_count,
        token_count * 2,
        token_count * 4,
        token_count * 8,
        token_count * 16
    ]

    benchmark_queries = [
        queries,
        queries * 2,
        queries * 4,
        queries * 8,
        queries * 16
    ]

    print("Token count:", token_count)
    print("Corpus length:", len(chopped_corpus))
    print("Number of queries:", len(queries))

    for i, (corpus, token_count, query_list) in enumerate(zip(benchmark_corpuss, benchmark_token_counts, benchmark_queries)):
        print(f"\nRunning benchmark for corpus size {token_count} and {len(query_list)} queries...")
        results, indexing_times = main(query_list, [corpus], k, [token_count])
        print("Benchmark Results:")
        print(results)
        print("Indexing Times:")
        print(indexing_times)
        results.to_csv(os.path.join(os.path.dirname(__file__), f"vlite5_benchmark_results_{i}.csv"), index=False)
        indexing_times.to_csv(os.path.join(os.path.dirname(__file__), f"vlite5_benchmark_indexing_times_{i}.csv"), index=False)
        
        