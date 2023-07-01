from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class EmbeddingModel:
    '''
    EmbeddingModel runs a transformer model and returns the embedding for a given text.
    '''
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.dimension = self.model.embeddings.position_embeddings.embedding_dim
        self.max_seq_length = self.model.embeddings.position_embeddings.num_embeddings\
        
        print("Dimension:", self.dimension)
        print("Max sequence length:", self.max_seq_length)
        
    
    def embed(self, text, max_seq_length=256):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        # print("Encoded input shape:", encoded_input['input_ids'].shape)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # print("Embeddings shape:", embeddings.shape)
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        embeddings = np.asarray([emb.numpy() for emb in embeddings])
        # print("Embeddings shape after np:", embeddings.shape)
    
        return embeddings