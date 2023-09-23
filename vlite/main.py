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
            
        self._collection = collection
        self._device = device
        self._model = EmbeddingModel() if model_name is None else EmbeddingModel(model_name)
        try:
            with np.load(self._collection, allow_pickle=True) as data:
                self._texts = data['texts'].tolist()
                self._metadata = data['metadata'].tolist()
                self._vectors = data['vectors']
        except FileNotFoundError:
            self._texts = []
            self._metadata = {}
            self._vectors = np.empty((0, self._model.dimension))
    
    def add_vector(self, vector):
        self._vectors = np.vstack((self._vectors, vector))

    def get_similar_vectors(self, vector, top_k=5, DEBUG=False):
        sims = cos_sim(vector, self._vectors)
        sims = sims[0]
        if DEBUG:
            print("[get_similar_vectors] Sims:", sims.shape)

        top_k_idx = np.argsort(sims)[::-1][:top_k]
        if DEBUG:
            print("[get_similar_vectors] Top k idx:", top_k_idx)
            print("[get_similar_vectors] Top k sims:", sims[top_k_idx])

        return top_k_idx, sims[top_k_idx]

    def memorize(self, text, id=None, metadata=None):
        id = id or str(uuid4())
        chunks = chop_and_chunk(text)
        encoded_data = self._model.embed(texts=chunks, device=self._device)
        self._vectors = np.vstack((self._vectors, encoded_data))
        for chunk in chunks:
            self._texts.append(chunk)
            idx = len(self._texts) - 1
            self._metadata[idx] = metadata or {}
            self._metadata[idx]['index'] = id or idx
        self.save()
        return id, self._vectors

    def remember(self, text=None, id=None, top_k=5):
        if id:
            return self._metadata[id]
        if text:

            sims = cos_sim(self._model.embed(texts=text, device=self._device) , self._vectors)
            print("[remember] Sims:", sims.shape)
            sims = sims[0]

			# top_k cannot be higher than the number of similarities returned
            top_k = min(top_k, len(sims))

            # Use np.argpartition to partially sort only the top k values
            top_k_idx = np.argpartition(sims, -top_k)[-top_k:]  

            # Use np.argsort to sort just those top k indices
            top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]  

            # print("[remember] Top k sims:", sims[top_k_idx])
            return [self._texts[idx] for idx in top_k_idx], sims[top_k_idx]
            
    def save(self):
        with open(self._collection, 'wb') as f:
            np.savez(f, texts=self._texts, metadata=self._metadata, vectors=self._vectors)

    @property
    def collection(self):
        return self._collection
    
    @property
    def device(self):
        return self._device
    
    @property
    def model(self):
        return self._model

    @property
    def texts(self):
        return self._texts
    
    @property
    def metadata(self):
        return self._metadata

    @property
    def vectors(self):
        return self._vectors