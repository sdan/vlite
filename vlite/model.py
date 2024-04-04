import os
import torch
import llama_cpp
from huggingface_hub import hf_hub_download
import tiktoken
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name='mixedbread-ai/mxbai-embed-large-v1'):
        hf_path = hf_hub_download(repo_id="mixedbread-ai/mxbai-embed-large-v1", filename="gguf/mxbai-embed-large-v1-f16.gguf")
        print(f"Downloaded model to {hf_path}")
        
        self.model = llama_cpp.Llama(model_path=hf_path, embedding=True)
        self.dimension = 1024 # hardcoded
        self.max_seq_length = 512 # hardcoded

    def embed(self, texts, max_seq_length=512, device="cpu"):
        if isinstance(texts, str):
            texts = [texts]
        embeddings_dict = self.model.create_embedding(texts)
        return [item["embedding"] for item in embeddings_dict["data"]]
    
    def token_count(self, texts):
        enc = tiktoken.get_encoding("cl100k_base")        
        tokens = 0
        for text in texts:
            token_ids = enc.encode(text, disallowed_special=())
            tokens += len(token_ids)
        return tokens
    
    def quantize(self, embeddings, precision="binary"):
        """
        Quantizes the embeddings to the specified precision.

        Args:
            embeddings (list or numpy.ndarray): Input embeddings to quantize.
            precision (str, optional): Precision to quantize the embeddings. Can be "binary" or "int8". Defaults to "binary".

        Returns:
            numpy.ndarray: Quantized embeddings.
        """
        embeddings = np.array(embeddings)  # Convert embeddings to a numpy array

        if precision == "binary":
            return np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1)
        elif precision == "int8":
            return ((embeddings - np.min(embeddings, axis=0)) / (np.max(embeddings, axis=0) - np.min(embeddings, axis=0)) * 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported precision: {precision}")

    def rescore(self, query_vector, vectors):
        """
        Rescores the retrieved vectors using the query vector.

        Args:
            query_vector (numpy.ndarray): Query vector for rescoring.
            vectors (numpy.ndarray): Retrieved vectors to rescore.

        Returns:
            numpy.ndarray: Rescored similarities.
        """
        return np.dot(query_vector, vectors.T).flatten()

    