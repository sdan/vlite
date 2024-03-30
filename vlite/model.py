import torch
from transformers import AutoModel, AutoTokenizer

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask, device="cpu"):
    device = torch.device(device)
    token_embeddings = model_output.last_hidden_state.to(device)
    attention_mask = attention_mask.to(device)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class EmbeddingModel:
    def __init__(self, model_name='mixedbread-ai/mxbai-embed-large-v1'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.dimension = self.model.embeddings.position_embeddings.embedding_dim
        self.max_seq_length = self.model.embeddings.position_embeddings.num_embeddings

    def embed(self, texts, max_seq_length=256, device="cpu"):
        device = torch.device(device)
        self.model.to(device)

        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_length)
        encoded_input = {name: tensor.to(device) for name, tensor in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'], device=device)
            tensor_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            np_embeddings = tensor_embeddings.cpu().numpy()

        return np_embeddings

    def token_count(self, texts):
        tokens = 0
        for text in texts:
            tokens += len(self.tokenizer.tokenize(text))
        return tokens