import numpy as np
from uuid import uuid4
from .model import EmbeddingModel
from .utils import chop_and_chunk, cos_sim
import datetime

class VLite:
    '''
    vlite is a simple vector database that stores vectors in a numpy array.
    '''
    def __init__(self, collection=None,device='mps',model_name=None):

		# Filename must be unique between runs. Saving to the same file will append vectors to previous run's vectors
        if collection is None:
            current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            collection = f"vlite_{current_datetime}.npz"
            
        self.collection = collection
        self.device = device
        self.model = EmbeddingModel() if model_name is None else EmbeddingModel(model_name)
        try:
            with np.load(self.collection, allow_pickle=True) as data:
                self.texts = data['texts'].tolist()
                self.metadata = data['metadata'].tolist()
                self.vectors = data['vectors']
        except FileNotFoundError:
            self.texts = []
            self.metadata = {}
            self.vectors = np.empty((0, self.model.dimension))
    
    def add_vector(self, vector):
        self.vectors = np.vstack((self.vectors, vector))

    def get_similar_vectors(self, vector, top_k=5):
        sims = cos_sim(vector, self.vectors)
        sims = sims[0]
        # print("[get_similar_vectors] Sims:", sims.shape)
        top_k_idx = np.argsort(sims)[::-1][:top_k]
        # print("[get_similar_vectors] Top k idx:", top_k_idx)
        # print("[get_similar_vectors] Top k sims:", sims[top_k_idx])
        return top_k_idx, sims[top_k_idx]

    def memorize(self, text, id=None, metadata=None):
        id = id or str(uuid4())
        chunks = chop_and_chunk(text)
        encoded_data = self.model.embed(texts=chunks, device=self.device)
        self.vectors = np.vstack((self.vectors, encoded_data))
        for chunk in chunks:
            self.texts.append(chunk)
            idx = len(self.texts) - 1
            self.metadata[idx] = metadata or {}
            self.metadata[idx]['index'] = id or idx
        self.save()
        return id, self.vectors

    def remember(self, text=None, id=None, top_k=5):
        if id:
            return self.metadata[id]
        if text:

            sims = cos_sim(self.model.embed(texts=text, device=self.device) , self.vectors)
            print("[remember] Sims:", sims.shape)
            sims = sims[0]

            # Use np.argpartition to partially sort only the top 5 values
            top_5_idx = np.argpartition(sims, -top_k)[-top_k:]  

            # Use np.argsort to sort just those top 5 indices
            top_5_idx = top_5_idx[np.argsort(sims[top_5_idx])[::-1]]  

            # print("[remember] Top k sims:", sims[top_5_idx])
            return [self.texts[idx] for idx in top_5_idx], sims[top_5_idx]
            
    def save(self):
        with open(self.collection, 'wb') as f:
            np.savez(f, texts=self.texts, metadata=self.metadata, vectors=self.vectors)
