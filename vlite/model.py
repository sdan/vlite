import os
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np
from typing import List
from tokenizers import Tokenizer
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import time


def normalize(v):
    if v.ndim == 1:
        v = v.reshape(1, -1)  # Reshape v to 2D array if it is 1D
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    return v / norm


def normalize_onnx(v):
    norm = np.linalg.norm(v, axis=1)
    norm[norm == 0] = 1e-12
    return v / norm[:, np.newaxis]



class EmbeddingModel:
    def __init__(self, model_name="mixedbread-ai/mxbai-embed-large-v1"):
        start_time = time.time()
        self.model = SentenceTransformer(model_name, device="mps")
        
        # tokenizer_path = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
        # model_path = hf_hub_download(repo_id=model_name, filename="onnx/model_fp16.onnx")

        # self.tokenizer = Tokenizer.from_file(tokenizer_path)
        # self.tokenizer.enable_truncation(max_length=512)
        # self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=512)

        # self.model = ort.InferenceSession(model_path)
        
        self.model_metadata = {
            "bert.embedding_length": 512,
            "bert.context_length": 512
        }
        self.embedding_size = self.model_metadata.get("bert.embedding_length", 1024)
        self.context_length = self.model_metadata.get("bert.context_length", 512)
        self.embedding_dtype = "float32"
        end_time = time.time()
        print(f"[model.__init__] Execution time: {end_time - start_time:.5f} seconds")
    
    def embed(self, texts, max_seq_length=512, device="mps", batch_size=32):
        start_time = time.time()
        if isinstance(texts, str):
            texts = [texts]  # Ensure texts is always a list
        embeddings = self.model.encode(texts, device=device, batch_size=batch_size, normalize_embeddings=True)
        end_time = time.time()
        print(f"[model.embed] Execution time: {end_time - start_time:.5f} seconds")
        return embeddings
    
    def encode_with_onnx(self, texts):
        # texts is a List[str]
    
        batch_size = 32
         # Ensure all text items are strings
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All items in the 'texts' list should be strings.")

        try:
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                encoded = [self.tokenizer.encode(d) for d in batch]
                input_ids = np.array([e.ids for e in encoded])
                attention_mask = np.array([e.attention_mask for e in encoded])
                onnx_input = {
                    "input_ids": np.array(input_ids, dtype=np.int64),
                    "attention_mask": np.array(attention_mask, dtype=np.int64),
                    "token_type_ids": np.array([np.zeros(len(e), dtype=np.int64) for e in input_ids], dtype=np.int64),
                }
                model_output = self.model.run(None, onnx_input)
                last_hidden_state = model_output[0]
                # Perform mean pooling with attention weighting
                input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), last_hidden_state.shape)
                embeddings = np.sum(last_hidden_state * input_mask_expanded, 1) / np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
                embeddings = normalize_onnx(embeddings).astype(np.float32)
                all_embeddings.append(embeddings)
            return np.concatenate(all_embeddings)
    
        except Exception as e:
            print(f"Failed during ONNX encoding: {e}")
            raise


    def quantize(self, embeddings, precision="binary"):
        start_time = time.time()
        # first normalize_embeddings to unit length
        embeddings = normalize(embeddings)
        # slice to get MRL embeddings
        embeddings_slice = embeddings[..., :512]
        
        if precision == "binary":
            end_time = time.time()
            print(f"[model.quantize] Execution time: {end_time - start_time:.5f} seconds")
            return self._binary_quantize(embeddings_slice)
        else:
            raise ValueError(f"Precision {precision} is not supported")

    def _binary_quantize(self, embeddings):
        return (np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1) - 128).astype(np.int8)

    def hamming_distance(self, embedding1, embedding2):
        # Ensure the embeddings are numpy arrays for the operation.
        return np.count_nonzero(np.array(embedding1) != np.array(embedding2))

    def search(self, query_embedding, embeddings, top_k):
        start_time = time.time()
        # Convert embeddings to a numpy array for efficient operations if not already.
        embeddings = np.array(embeddings)
        distances = np.array([self.hamming_distance(query_embedding, emb) for emb in embeddings])
        
        # Find the indices of the top_k smallest distances
        top_k_indices = np.argsort(distances)[:top_k]
        end_time = time.time()
        print(f"[model.search] Execution time: {end_time - start_time:.5f} seconds")
        return top_k_indices, distances[top_k_indices]