import torch
from transformers import AutoModel, AutoTokenizer


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask, device="mps"):
    device = torch.device(device)  # Create a torch.device object for the MPS device
    token_embeddings = model_output.last_hidden_state.to(device)  # Move token_embeddings to MPS device
    attention_mask = attention_mask.to(device)  # Move attention_mask to MPS device
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class EmbeddingModel:
    '''
    EmbeddingModel runs a transformer model and returns the embedding for a given text.
    '''

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # use_fast=True

        self.model = AutoModel.from_pretrained(model_name)
        self.dimension = self.model.embeddings.position_embeddings.embedding_dim
        self.max_seq_length = self.model.embeddings.position_embeddings.num_embeddings

    def embed(self, texts, max_seq_length=256, device="mps"):
        if (torch.backends.mps.is_available()):
            dev = torch.device("mps")
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        device = torch.device(dev)  # Create a torch.device object
        self.model.to(device)  # Move the model to the specified device

        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt',
                                       max_length=max_seq_length)

        if encoded_input['input_ids'].shape[0] > 1300:
            device = torch.device("cpu")
            self.model.to(device)  # Move the model to the specified device

        encoded_input = {name: tensor.to(device) for name, tensor in
                         encoded_input.items()}  # Move all input tensors to the specified device

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embeddings = mean_pooling(model_output, encoded_input['attention_mask'], device=device)
        tensor_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        np_embeddings = tensor_embeddings.cpu().numpy()  # Move tensor to CPU before converting to numpy

        return np_embeddings

    def token_count(self, texts):
        tokens = 0
        for text in texts:
            tokens += len(self.tokenizer.tokenize(text))