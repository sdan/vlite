import numpy as np
from transformers import AutoModel, AutoTokenizer
import time
import torch
from typing import Dict

# def normalize(v):
#     return torch.nn.functional.normalize(v, p=2, dim=1)



def transform_query(query: str) -> str:
    return f'Represent this sentence for searching relevant passages: {query}'

def pooling(outputs: torch.Tensor, inputs: Dict,  strategy: str = 'cls') -> np.ndarray:
    if strategy == 'cls':
        pooled_output = outputs[:, 0]
    elif strategy == 'mean':
        attention_mask = inputs["attention_mask"][:, :, None]
        sum_embeddings = torch.sum(outputs * attention_mask, dim=1)
        sum_mask = torch.sum(attention_mask, dim=1)
        pooled_output = sum_embeddings / sum_mask
    else:
        raise NotImplementedError
    return pooled_output


def cos_sim_np(a, b):
    # Ensure both a and b are at least 2D
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    dot_product = np.dot(a, b.T)
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    return (dot_product / (norm_a * norm_b)).flatten()  # Flatten to handle cases where we compare just two vectors


class EmbeddingModel:
    def __init__(self, model_name="mixedbread-ai/mxbai-embed-large-v1", device='cpu'):
        start_time = time.time()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model_metadata = {
            "bert.embedding_length": 512,
            "bert.context_length": 512
        }
        self.embedding_size = self.model_metadata.get("bert.embedding_length", 1024)
        self.context_length = self.model_metadata.get("bert.context_length", 512)
        self.embedding_dtype = "float32"
        end_time = time.time()
        print(f"[model.__init__] Execution time: {end_time - start_time:.5f} seconds")

    def embed(self, texts, batch_size=32, precision="binary"):
        start_time = time.time()
        if isinstance(texts, str):
            texts = [texts]  # Ensure texts is always a list

        # Tokenize and prepare inputs
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.context_length, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Perform the forward pass
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
            embeddings = pooling(outputs, inputs, 'cls')

        # Normalize embeddings to unit length
        assert isinstance(embeddings, torch.Tensor), "Embeddings must be a PyTorch tensor"
        embeddings = self.normalize(embeddings)

        # Slice to get MRL embeddings
        embeddings_slice = embeddings[..., :512]

        if precision == "binary":
            binary_embeddings = (embeddings_slice > 0).byte()
            quantized_embeddings = binary_embeddings.cpu().numpy()
            np.packbits(quantized_embeddings, axis=-1)
            end_time = time.time()
            print(f"[model.embed] Execution time: {end_time - start_time:.5f} seconds")
            print(f"Quantized embeddings shape: {quantized_embeddings.shape}")
            return quantized_embeddings
        else:
            raise ValueError(f"Precision {precision} is not supported")

    def hamming_distance(self, embedding1, embedding2):
        # Calculate Hamming distance directly using bitwise operations
        return np.sum(embedding1 != embedding2, axis=1)

    def search(self, query_embedding, embeddings, top_k):
        hamming_distances = np.sum(embeddings != query_embedding, axis=1)
        max_distance = embeddings.shape[1] * 8  # Maximum possible Hamming distance
        top_k_indices = np.argsort(hamming_distances)[:top_k]
        top_k_scores = hamming_distances[top_k_indices]
        top_k_scores = 1 - hamming_distances / max_distance
        return top_k_indices, top_k_scores

    def normalize(self, v):
        if isinstance(v, np.ndarray):
            # Convert to tensor if accidentally given a numpy array (this should not happen if data flow is correct)
            v = torch.tensor(v, dtype=torch.float32, device=self.device)
        return torch.nn.functional.normalize(v, p=2, dim=1)


    def token_count(self, text):
        encoded_input = self.tokenizer.encode_plus(text, return_tensors='pt')
        return len(encoded_input['input_ids'][0])