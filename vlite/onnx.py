# A dependency-light way to run the onnx model

from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np
from typing import List

MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"

# Use pytorches default epsilon for division by zero
# https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
def normalize(v):
    norm = np.linalg.norm(v, axis=1)
    norm[norm == 0] = 1e-12
    return v / norm[:, np.newaxis]

# Sampel implementation of the default sentence-transformers model using ONNX
class ONNXModel():

    def __init__(self):
        # max_seq_length = 256, for some reason sentence-transformers uses 256 even though the HF config has a max length of 128
        # https://github.com/UKPLab/sentence-transformers/blob/3e1929fddef16df94f8bc6e3b10598a98f46e62d/docs/_static/html/models_en_sentence_embeddings.html#LL480
        self.tokenizer = Tokenizer.from_file("onnx/tokenizer.json")
        print("[tokenizer ]",self.tokenizer.get_vocab_size())
        print("[tokenizer ]",self.tokenizer.get_vocab())

        
        self.tokenizer.enable_truncation(max_length=512)
        self.model = ort.InferenceSession("onnx/model.onnx")
        print("[model ]",self.model.get_modelmeta())
        

    def __call__(self, documents: List[str], batch_size: int = 32):
        all_embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
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
            embeddings = normalize(embeddings).astype(np.float32)
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings)


# sample_text = "This is a sample text that is likely to overflow the entire model and will be truncated. \
#     Keep writing and writing until we reach the end of the model.This is a sample text that is likely to overflow the entire model and \
#     will be truncated. Keep writing and writing until we reach the end of the model.This is a sample text that is likely to overflow the entire \
#     model and will be truncated. Keep writing and writing until we reach the end of the model. This is a sample text that is likely to overflow \
#     the entire model and will be truncated. Keep writing and writing until we reach the end of the model. This is a sample text that is likely to overflow  \
#     the entire model and will be truncated. Keep writing and writing until we reach the end of the model."
# model = DefaultEmbeddingModel()
# # print(model([sample_text, sample_text]))

# embeddings = model([sample_text, sample_text])
# print(embeddings.shape)
# # print(embeddings[0] == embeddings[1])
# # print(embeddings)