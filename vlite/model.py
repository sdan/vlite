import numpy as np
from transformers import AutoModel, AutoTokenizer
import time

def normalize(v):
    if v.ndim == 1:
        v = v.reshape(1, -1)  # Reshape v to 2D array if it is 1D
    # Detach the tensor if it's a PyTorch tensor and requires grad
    v = v.detach().cpu().numpy() if hasattr(v, 'detach') else v
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    return v / norm


def transform_query(query: str) -> str:
    return f'Represent this sentence for searching relevant passages: {query}'


def pooling_np(outputs, attention_mask, strategy='cls'):
    if strategy == 'cls':
        # Taking the first token (CLS token) for each sequence
        return outputs[:, 0]
    elif strategy == 'mean':
        # Applying attention mask and computing mean pooling
        outputs_masked = outputs * attention_mask[:, :, None]
        return np.sum(outputs_masked, axis=1) / np.sum(attention_mask, axis=1)[:, None]
    else:
        raise NotImplementedError


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
    def __init__(self, model_name="mixedbread-ai/mxbai-embed-large-v1"):
        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.model_metadata = {
            "bert.embedding_length": 512,
            "bert.context_length": 512
        }
        self.embedding_size = self.model_metadata.get("bert.embedding_length", 1024)
        self.context_length = self.model_metadata.get("bert.context_length", 512)
        self.embedding_dtype = "float32"
        end_time = time.time()
        print(f"[model.__init__] Execution time: {end_time - start_time:.5f} seconds")
    
    def embed(self, texts, device='cpu', batch_size=32):
        start_time = time.time()
        if isinstance(texts, str):
            texts = [texts]  # Ensure texts is always a list

        # Tokenize and prepare inputs
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.model.config.max_position_embeddings, return_tensors='pt')
        
        # Explicitly move tensors to the device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        # Move model to the device
        self.model.to(device)

        # Perform the forward pass
        outputs = self.model(**inputs).last_hidden_state
        attention_mask = inputs['attention_mask']

        # Pool embeddings using your numpy-based pooling method
        embeddings = pooling_np(outputs.detach().cpu().numpy(), attention_mask.detach().cpu().numpy(), 'cls')

        end_time = time.time()
        print(f"[model.embed] Execution time: {end_time - start_time:.5f} seconds")
        return embeddings




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
        # Calculate Hamming distance directly using bitwise operations
        return np.count_nonzero(np.bitwise_xor(embedding1, embedding2))

    def search(self, query_embedding, embeddings, top_k):
        query_embedding = np.atleast_2d(query_embedding)

        # Remove the type check for boolean, as we handle integer packed binaries
        distances = np.array([self.hamming_distance(query_embedding.flatten(), emb.flatten()) for emb in embeddings])

        # Find the indices of the top_k smallest distances
        top_k_indices = np.argsort(distances)[:top_k]
        top_k_scores = distances[top_k_indices]

        return top_k_indices, top_k_scores


