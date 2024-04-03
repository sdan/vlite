import os
import torch
import llama_cpp
from huggingface_hub import hf_hub_download

class EmbeddingModel:
    def __init__(self, model_name='mixedbread-ai/mxbai-embed-large-v1'):
        hf_path = hf_hub_download(repo_id="mixedbread-ai/mxbai-embed-large-v1", filename="gguf/mxbai-embed-large-v1-f16.gguf")
        print(f"Downloaded model to {hf_path}")
        
        self.model = llama_cpp.Llama(model_path=hf_path, embedding=True)
        self.dimension = 1024 # hardcoded
        self.max_seq_length = 512 # hardcoded

    def embed(self, texts, max_seq_length=512, device="cpu"):
        embeddings_dict = self.model.create_embedding(texts)
        return [item["embedding"] for item in embeddings_dict["data"]]
    
    def token_count(self, texts):
        tokens = 0
        for text in texts:
            tokens += len(self.tokenizer.tokenize(text))
        return tokens
    