import numpy as np
from sentence_transformers import SentenceTransformer
from usearch.index import Index, Matches
from usearch.compiled import ScalarKind
import datetime
import os
from typing import List, Literal, Tuple, Dict, Optional, Union
import logging
import math
import time


logger = logging.getLogger(__name__)

def chop_and_chunk(text: str, max_seq_length: int = 512) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= max_seq_length:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

class VLite:
    def __init__(self, vdb_name: str = None, device: str = 'cpu', embedding_model: str = 'mixedbread-ai/mxbai-embed-large-v1'):
        vdb_name = vdb_name if vdb_name is not None else f"vlite_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        vdb_name = os.path.splitext(vdb_name)[0] if vdb_name is not None else vdb_name

        self.__name = vdb_name
        self.__metadata_file = f"{vdb_name}.info"
        self.__index_file = f"{vdb_name}.index"

        metadata_exists = os.path.exists(self.__metadata_file)
        index_exists = os.path.exists(self.__index_file)
        if metadata_exists != index_exists:
            raise Exception("Must have BOTH .info and .index file present. Cannot continue unless neither or both files exist.")

        self.device = device
        self.__embed_model = SentenceTransformer(embedding_model)

        self.__index: Index = Index(ndim=1024, metric='cos', dtype='f32', path=self.__index_file)  # existence handled within USearch Index Object

        if metadata_exists:
            with np.load(self.__metadata_file, allow_pickle=True) as data:
                self.__texts = data['texts'].tolist()
                self.__metadata = data['metadata'].tolist()
                self.__chunk_id = int(data['chunk_id'])
                self.__document_id = int(data['document_id'])
        else:
            self.__texts = {}
            self.__metadata = {}
            self.__chunk_id = 0
            self.__document_id = 0

    def ingest(self, text: str, max_seq_length: int = 512, metadata: dict = None) -> int:
        if not isinstance(text, str):
            raise TypeError("The 'text' argument must be a string.")
        if metadata and not isinstance(metadata, dict):
            raise TypeError("The 'metadata' argument must be a dict.")

        chunks = chop_and_chunk(text, max_seq_length=max_seq_length)
        encoded_chunks = self.__embed_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

        binary_chunks = quantize_embeddings(encoded_chunks, precision='binary')  # quantize embeddings to binary
        int8_chunks = quantize_embeddings(encoded_chunks, precision='int8')  # quantize embeddings to int8

        keys = np.arange(self.__chunk_id, self.__chunk_id + len(chunks))
        self.__index.add(keys=keys, vectors=binary_chunks)  # add binary embeddings to index

        for i, chunk in enumerate(chunks):
            self.__texts[self.__chunk_id] = chunk
            self.__metadata[self.__chunk_id] = metadata or {}
            self.__metadata[self.__chunk_id]['document_id'] = self.__document_id
            self.__metadata[self.__chunk_id]['int8_embedding'] = int8_chunks[i]  # store int8 embeddings in metadata
            self.__chunk_id += 1

        self.__document_id += 1
        self.save()

        return self.__document_id - 1


    def retrieve(self, text: str, top_k: int = 3, rescore_multiplier: int = 10, autocut: bool = False, autocut_amount: int = 25, get_metadata: bool = False, get_similarities: bool = False, progress: bool = False) -> dict:
        if not isinstance(text, str):
            raise TypeError("The 'text' argument must be a string.")
        if top_k <= 0:
            raise Exception("Please input k >= 1.")

        query_embedding = self.__embed_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        query_float = query_embedding.astype(np.float32)  # keep float32 query for rescoring
        query_binary = quantize_embeddings(query_embedding, precision='binary')

        # Perform initial retrieval using binary embeddings
        results, search_time = semantic_search_usearch(
            query_embeddings=query_binary,
            corpus_index=self.__index,
            corpus_precision="binary",
            top_k=top_k * rescore_multiplier,
            rescore=False,
            exact=True,
            output_index=False,
        )

        # Rescore the retrieved documents using int8 embeddings
        candidate_indices = [result["corpus_id"] for result in results[0]]
        candidate_int8_embeddings = [self.__metadata[idx]['int8_embedding'] for idx in candidate_indices]
        rescored_scores = np.dot(query_float, np.array(candidate_int8_embeddings).T)

        # Sort the rescored documents and return the top-k
        sorted_indices = np.argsort(rescored_scores)[::-1][:top_k]
        final_indices = [candidate_indices[idx] for idx in sorted_indices]
        final_scores = [rescored_scores[idx] for idx in sorted_indices]

        if autocut:
            sim_diff = np.diff(final_scores)
            std_dev = np.std(sim_diff)
            cluster_idxs = np.where(sim_diff > std_dev)[0]  # indices marking the position of the END of each autocut cluster
            k = min(top_k, len(cluster_idxs))  # ensures that only pull a MAX of top_k clusters (can have less if no other relevant clusters found)
            if cluster_idxs.size > 0:
                endpoint = cluster_idxs[k - 1] + 1  # gets indices of elements in top k CLUSTERS
            else:
                endpoint = k
            final_indices = final_indices[0:endpoint]
            final_scores = final_scores[0:endpoint]

        texts: list = [self.__texts[idx] for idx in final_indices]

        results = {"texts": texts}
        if get_metadata:
            metadata: list = [self.__metadata[idx] for idx in final_indices]
            results["metadata"] = metadata
        if get_similarities:
            results["scores"] = final_scores
        return results


def semantic_search_usearch(
    query_embeddings: np.ndarray,
    corpus_embeddings: Optional[np.ndarray] = None,
    corpus_index: Optional["usearch.index.Index"] = None,
    corpus_precision: Literal["float32", "int8", "binary"] = "float32",
    top_k: int = 10,
    ranges: Optional[np.ndarray] = None,
    calibration_embeddings: Optional[np.ndarray] = None,
    rescore: bool = True,
    rescore_multiplier: int = 2,
    exact: bool = True,
    output_index: bool = False,
) -> Tuple[List[List[Dict[str, Union[int, float]]]], float, "usearch.index.Index"]:
    from usearch.index import Index
    from usearch.compiled import ScalarKind

    if corpus_embeddings is not None and corpus_index is not None:
        raise ValueError("Only corpus_embeddings or corpus_index should be used, not both.")
    if corpus_embeddings is None and corpus_index is None:
        raise ValueError("Either corpus_embeddings or corpus_index should be used.")
    if corpus_precision not in ["float32", "int8", "binary"]:
        raise ValueError('corpus_precision must be "float32", "int8", or "binary" for usearch')

    # If corpus_index is not provided, create a new index
    if corpus_index is None:
        if corpus_precision == "float32":
            corpus_index = Index(
                ndim=corpus_embeddings.shape[1],
                metric="cos",
                dtype="f32",
            )
        elif corpus_precision == "int8":
            corpus_index = Index(
                ndim=corpus_embeddings.shape[1],
                metric="ip",
                dtype="i8",
            )
        elif corpus_precision == "binary":
            corpus_index = Index(
                ndim=corpus_embeddings.shape[1],
                metric="hamming",
                dtype="i8",
            )
        corpus_index.add(np.arange(len(corpus_embeddings)), corpus_embeddings)

    # If rescoring is enabled and the query embeddings are in float32, we need to quantize them
    # to the same precision as the corpus embeddings. Also update the top_k value to account for the
    # rescore_multiplier
    rescore_embeddings = None
    k = top_k
    if query_embeddings.dtype not in (np.uint8, np.int8):
        if rescore:
            if corpus_index.dtype != ScalarKind.F32:
                rescore_embeddings = query_embeddings
                k *= rescore_multiplier
            else:
                logger.warning(
                    "Rescoring is enabled but the corpus is not quantized. Either pass `rescore=False` or "
                    'quantize the corpus embeddings with `quantize_embeddings(embeddings, precision="...") `'
                    'and pass `corpus_precision="..."` to `semantic_search_usearch`.'
                )

        query_embeddings = quantize_embeddings(
            query_embeddings,
            precision=corpus_precision,
            ranges=ranges,
            calibration_embeddings=calibration_embeddings,
        )
    elif rescore:
        logger.warning(
            "Rescoring is enabled but the query embeddings are quantized. Either pass `rescore=False` or don't quantize the query embeddings."
        )

    # Perform the search using the usearch index
    start_t = time.time()
    matches = corpus_index.search(query_embeddings, count=k, exact=exact)
    scores = matches.distances
    indices = matches.keys

    # If rescoring is enabled, we need to rescore the results using the rescore_embeddings
    if rescore_embeddings is not None:
        top_k_embeddings = np.array([corpus_index.get(query_indices) for query_indices in indices])
        # If the corpus precision is binary, we need to unpack the bits
        if corpus_precision == "binary":
            top_k_embeddings = np.unpackbits(top_k_embeddings.astype(np.uint8), axis=-1)
        top_k_embeddings = top_k_embeddings.astype(int)

        # rescore_embeddings: [num_queries, embedding_dim]
        # top_k_embeddings: [num_queries, top_k, embedding_dim]
        # updated_scores: [num_queries, top_k]
        # We use einsum to calculate the dot product between the query and the top_k embeddings, equivalent to looping
        # over the queries and calculating 'rescore_embeddings[i] @ top_k_embeddings[i].T'
        rescored_scores = np.einsum("ij,ikj->ik", rescore_embeddings, top_k_embeddings)
        rescored_indices = np.argsort(-rescored_scores)[:, :top_k]
        indices = indices[np.arange(len(query_embeddings))[:, None], rescored_indices]
        scores = rescored_scores[np.arange(len(query_embeddings))[:, None], rescored_indices]

    delta_t = time.time() - start_t

    outputs = (
        [
            [
                {"corpus_id": int(neighbor), "score": float(score)}
                for score, neighbor in zip(scores[query_id], indices[query_id])
            ]
            for query_id in range(len(query_embeddings))
        ],
        delta_t,
    )
    if output_index:
        outputs = (*outputs, corpus_index)
    return outputs


def quantize_embeddings(
    embeddings: np.ndarray,
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"],
    ranges: Optional[np.ndarray] = None,
    calibration_embeddings: Optional[np.ndarray] = None,
) -> np.ndarray:
    if embeddings.dtype in (np.uint8, np.int8):
        raise Exception("Embeddings to quantize must be float rather than int8 or uint8.")

    if precision == "float32":
        return embeddings.astype(np.float32)

    if precision.endswith("int8"):
        # Either use the 1. provided ranges, 2. the calibration dataset or 3. the provided embeddings
        if ranges is None:
            if calibration_embeddings is not None:
                ranges = np.vstack((np.min(calibration_embeddings, axis=0), np.max(calibration_embeddings, axis=0)))
            else:
                if embeddings.shape[0] < 100:
                    logger.warning(
                        f"Computing {precision} quantization buckets based on {len(embeddings)} embedding{'s' if len(embeddings) != 1 else ''}."
                        f" {precision} quantization is more stable with `ranges` calculated from more embeddings "
                        "or a `calibration_embeddings` that can be used to calculate the buckets."
                    )
                ranges = np.vstack((np.min(embeddings, axis=0), np.max(embeddings, axis=0)))
        starts = ranges[0, :]
        steps = (ranges[1, :] - ranges[0, :]) / 255

        if precision == "uint8":
            return ((embeddings - starts) / steps).astype(np.uint8)
        elif precision == "int8":
            return ((embeddings - starts) / steps - 128).astype(np.int8)

    if precision == "binary":
        return (np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1) - 128).astype(np.int8)

    if precision == "ubinary":
        return np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1)

    raise ValueError(f"Precision {precision} is not supported")