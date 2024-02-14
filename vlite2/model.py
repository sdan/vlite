import torch
from transformers import AutoModel, AutoTokenizer

def mean_pooling(model_output, attention_mask, device="mps"):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings).to(device, non_blocking=True).float()
    sum_embeddings = torch.sum(token_embeddings.to(device, non_blocking=True) * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1).clamp_min(1e-9)
    return sum_embeddings / sum_mask

class EmbeddingModel:
    '''
    EmbeddingModel runs a transformer model and returns the embedding for a given text.
    '''
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{model_name}", use_fast=True)

        self.model = AutoModel.from_pretrained(f"sentence-transformers/{model_name}")
        self.dimension = self.model.embeddings.position_embeddings.embedding_dim
        self.max_seq_length = self.model.embeddings.position_embeddings.num_embeddings

    def embed(self, texts, max_seq_length=256, device="mps"):
        if(torch.backends.mps.is_available()):
            dev = torch.device("mps")
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        device = torch.device(dev)
        self.model.to(device)

        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_length)
        
        if encoded_input['input_ids'].shape[0] > 1300:
            device = torch.device("cpu")
            self.model.to(device)
        
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
            tokens+=len(self.tokenizer.tokenize(text))
    