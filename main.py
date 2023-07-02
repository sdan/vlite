import numpy as np
import pickle
from uuid import uuid4
from model import EmbeddingModel
from utils import chop_and_chunk, cos_sim

class VLite:
    '''
    vlite is a simple vector database that stores vectors in a numpy array.
    '''
    def __init__(self, collection='vlite.pkl'):
        self.collection = collection
        self.model = EmbeddingModel()
        try:
            with open(self.collection, 'rb') as f:
                self.texts, self.metadata, self.vectors = pickle.load(f)
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
        # print("[+] Chunks:", chunks)
        for chunk in chunks:
            encoded_data = self.model.embed(chunk)
            self.texts.append(chunk)
            self.metadata[len(self.texts) - 1] = metadata or {}
            self.metadata[len(self.texts) - 1]['index'] = id or len(self.texts) - 1
            self.vectors = np.vstack((self.vectors, encoded_data)) 

        self.save()
        return id, self.vectors

    def remember(self, text=None, id=None, top_k=5):
        if id:
            return self.metadata[id]
        if text:

            sims = cos_sim(self.model.embed(text) , self.vectors)
            sims = sims[0]

            # Use np.argpartition to partially sort only the top 5 values
            top_5_idx = np.argpartition(sims, -top_k)[-top_k:]  

            # Use np.argsort to sort just those top 5 indices
            top_5_idx = top_5_idx[np.argsort(sims[top_5_idx])[::-1]]  

            # print("[remember] Top k sims:", sims[top_5_idx])
            return [self.texts[idx] for idx in top_5_idx], sims[top_5_idx]
            
    def save(self):
        with open(self.collection, 'wb') as f:
            pickle.dump((self.texts, self.metadata, self.vectors), f)