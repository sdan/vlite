import os
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np
from typing import List
from tokenizers import Tokenizer
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


def normalize(v):
    if v.ndim == 1:
        v = v.reshape(1, -1)  # Reshape v to 2D array if it is 1D
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    return v / norm



class EmbeddingModel:
    def __init__(self, model_name="mixedbread-ai/mxbai-embed-large-v1"):
        self.model = SentenceTransformer(model_name)
        self.model_metadata = {
            "bert.embedding_length": 512,
            "bert.context_length": 512
        }
        self.embedding_size = self.model_metadata.get("bert.embedding_length", 1024)
        self.context_length = self.model_metadata.get("bert.context_length", 512)
        self.embedding_dtype = "float32"
    
    def embed(self, texts, max_seq_length=512, device="cpu", batch_size=32):
        if isinstance(texts, str):
            texts = [texts]  # Ensure texts is always a list
        embeddings = self.model.encode(texts, device=device, batch_size=batch_size, normalize_embeddings=True)
        return embeddings


    def quantize(self, embeddings, precision="binary"):
        # first normalize_embeddings to unit length
        embeddings = normalize(embeddings)
        # slice to get MRL embeddings
        embeddings_slice = embeddings[..., :512]
        
        if precision == "binary":
            return self._binary_quantize(embeddings_slice)
        else:
            raise ValueError(f"Precision {precision} is not supported")

    def _binary_quantize(self, embeddings):
        return (np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1) - 128).astype(np.int8)

    def hamming_distance(self, embedding1, embedding2):
        # Ensure the embeddings are numpy arrays for the operation.
        return np.count_nonzero(np.array(embedding1) != np.array(embedding2))

    def search(self, query_embedding, embeddings, top_k):
        # Convert embeddings to a numpy array for efficient operations if not already.
        embeddings = np.array(embeddings)
        distances = np.array([self.hamming_distance(query_embedding, emb) for emb in embeddings])
        
        # Find the indices of the top_k smallest distances
        top_k_indices = np.argsort(distances)[:top_k]
        return top_k_indices, distances[top_k_indices]