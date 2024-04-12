import os
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np
from typing import List
from tokenizers import Tokenizer

def normalize(v):
    norm = np.linalg.norm(v, axis=1)
    norm[norm == 0] = 1e-12
    return v / norm[:, np.newaxis]

class EmbeddingModel:
    def __init__(self, model_name="mixedbread-ai/mxbai-embed-large-v1"):
        tokenizer_path = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
        model_path = hf_hub_download(repo_id=model_name, filename="onnx/model.onnx")
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=512)
        
        self.model = ort.InferenceSession(model_path)
        print("[model]", self.model.get_modelmeta())
        
        self.model_metadata = {
            "bert.embedding_length": 1024,
            "bert.context_length": 512
        }
        self.embedding_size = self.model_metadata.get("bert.embedding_length", 1024)
        self.context_length = self.model_metadata.get("bert.context_length", 512)
        self.embedding_dtype = "float32"

    def embed(self, texts: List[str], max_seq_length=512, device="cpu", batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = [self.tokenizer.encode(d) for d in batch]
            input_ids = np.array([e.ids for e in encoded])
            attention_mask = np.array([e.attention_mask for e in encoded])
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

            onnx_input = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            model_output = self.model.run(None, onnx_input)
            last_hidden_state = model_output[0]

            input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), last_hidden_state.shape)
            embeddings = np.sum(last_hidden_state * input_mask_expanded, 1) / np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
            embeddings = normalize(embeddings).astype(np.float32)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings)

    def token_count(self, texts):
        tokens = 0
        for text in texts:
            encoded = self.tokenizer.encode(text)
            tokens += len(encoded.ids)
        return tokens

    def quantize(self, embeddings, precision="binary", ranges=None, calibration_embeddings=None):
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        if precision == "binary":
            # Quantize embeddings to binary by thresholding at 0
            quantized_embeddings = (embeddings > 0).astype(np.uint8)
        elif precision == "int8":
            if ranges is None:
                # Compute ranges from calibration embeddings if not provided
                if calibration_embeddings is None:
                    calibration_embeddings = embeddings
                ranges = np.stack([np.min(calibration_embeddings, axis=0), np.max(calibration_embeddings, axis=0)])

            # Quantize embeddings to int8 using the computed ranges
            quantized_embeddings = np.clip(np.round((embeddings - ranges[0]) / (ranges[1] - ranges[0]) * 255), 0, 255).astype(np.int8)
        else:
            raise ValueError(f"Unsupported precision: {precision}")

        return quantized_embeddings